"""Context-scoped stage timing and memory profiling."""

import os
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

import torch


def profiling_output_dir(conf, args, *, operation: str = "benchmark") -> Path:
    """Return the owned profiling artifact directory for one selected target."""
    from vidmap.configuration.names import config_name_to_output_slug

    if hasattr(args, "name"):
        profile_name = args.name
    else:
        config_name = args.frontend_conf if operation == "frontend" else args.mapping_conf
        derived_name = config_name if operation == "frontend" else config_name_to_output_slug(config_name)
        custom_name = conf.name != derived_name
        profile_name = conf.name if custom_name else config_name.replace("/", "_")
    scene = conf.selection.scene[0] if conf.selection.scene else "default"
    testset = conf.selection.testset_id[0] if conf.selection.testset_id else "default"
    return Path("profiling") / operation / args.dataset / profile_name / scene / testset / conf.selection.mode


@dataclass
class ProfilingSession:
    """Measurements owned by one explicitly profiled process invocation."""

    enabled: bool
    timings: OrderedDict[str, float] = field(default_factory=OrderedDict)
    memory: OrderedDict[str, float] = field(default_factory=OrderedDict)

    def record_timing(self, stage: str, seconds: float, *, first: bool = False) -> None:
        if not self.enabled or (first and stage in self.timings):
            return
        self.timings[stage] = seconds

    def record_memory(self, stage: str, rss_gb: float) -> None:
        if self.enabled:
            self.memory[stage] = rss_gb

    def write(self, path: str | Path) -> None:
        if not self.enabled:
            raise RuntimeError("cannot write a disabled profiling session")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wall_total = self.timings["wall_total"] if "wall_total" in self.timings else None
        with path.open("w") as file:
            for name, duration in self.timings.items():
                if wall_total is None:
                    file.write(f"{name}: {duration:.2f}s\n")
                else:
                    percent = 100 * duration / wall_total if wall_total > 0 else 0
                    file.write(f"{name}: {duration:.2f}s ({percent:.1f}%)\n")
            if wall_total is not None:
                file.write(f"TOTAL: {wall_total:.2f}s\n")
            if self.memory:
                file.write("\n")
                for stage, rss in self.memory.items():
                    file.write(f"mem_{stage}: {rss:.2f} GB\n")
                file.write(f"mem_sampled_rss_max: {max(self.memory.values()):.2f} GB\n")


_ACTIVE_SESSION: ContextVar[ProfilingSession | None] = ContextVar("vidmap_profiling_session", default=None)


@contextmanager
def profiling_session(enabled: bool) -> Iterator[ProfilingSession]:
    """Create an isolated profiling collector for one run."""
    session = ProfilingSession(enabled=enabled)
    token = _ACTIVE_SESSION.set(session)
    try:
        yield session
    finally:
        _ACTIVE_SESSION.reset(token)


def _active_enabled_session() -> ProfilingSession | None:
    session = _ACTIVE_SESSION.get()
    return session if session is not None and session.enabled else None


def sync_time():
    if _active_enabled_session() is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def record_timing(stage: str, seconds: float, *, first: bool = False) -> None:
    session = _active_enabled_session()
    if session is not None:
        session.record_timing(stage, seconds, first=first)


def get_rss_gb():
    """Current process RSS in GB via /proc/self/statm (no imports needed)."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    with open("/proc/self/statm") as f:
        rss_pages = int(f.read().split()[1])
    return rss_pages * page_size / (1024**3)


def log_memory(stage):
    session = _active_enabled_session()
    if session is None:
        return
    rss = get_rss_gb()
    session.record_memory(stage, rss)

"""Checksummed, atomic storage around PyTorch's compiled-function serializer."""

import fcntl
import hashlib
import inspect
import json
import logging
import os
import pickle
import platform
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch.utils._pytree import tree_flatten, treespec_dumps

from vidmap.frontend.cache import file_fingerprint

logger = logging.getLogger(__name__)


def tensor_signature(value):
    return {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def runtime_identity():
    import triton
    from torch._inductor import config

    device = torch.cuda.current_device()
    compiler_config = pickle.loads(config.save_config())
    # CUDA graphs do not depend on the XPU path, which defaults to the launch directory.
    compiler_config.pop("xpu.cutlass_dir", None)
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "device": device,
        "inductor": hashlib.sha256(pickle.dumps(compiler_config, protocol=2)).hexdigest(),
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"] if "CUBLAS_WORKSPACE_CONFIG" in os.environ else None,
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
        "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
    }


@contextmanager
def graph_entry(root, key):
    """Lock an entry and publish a successful capture atomically."""
    root.mkdir(parents=True, exist_ok=True)
    directory = root / key
    with (root / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if directory.exists():
            logger.info(f"Loading saved graph: {directory}")
            yield directory, False
        else:
            logger.info(f"Capturing graph: {directory}")
            with TemporaryDirectory(prefix=f".{key}-", dir=root) as temporary:
                candidate = Path(temporary) / "entry"
                candidate.mkdir()
                yield candidate, True
                candidate.rename(directory)


def _manifest(directory, identity, payload, *, write=False):
    path = directory / "manifest.json"
    if write:
        path.write_text(json.dumps({"identity": identity, "sha256": file_fingerprint(directory / payload)}) + "\n")
    else:
        manifest = json.loads(path.read_text())
        if manifest["identity"] != identity or manifest["sha256"] != file_fingerprint(directory / payload):
            raise ValueError(f"Compiled artifact identity/checksum mismatch: {directory}; remove it to rebuild")


class CachedGraph(torch.nn.Module):
    """Share model weights across shapes; delegate executable persistence to PyTorch."""

    def __init__(self, *, net=None, factory=None, namespace, component, model_identity, sources, extra_identity=None):
        super().__init__()
        assert (net is None) != (factory is None)
        self.net = net
        self._factory = factory
        self.graphs = {}
        self.persistent = (
            "load_compiled_function" in vars(torch.compiler)
            and "f_globals" in inspect.signature(torch.compiler.load_compiled_function).parameters
        )
        if not self.persistent:
            logger.info("Compiled-function persistence is unavailable; using ordinary torch.compile")
        self.runtime = runtime_identity()
        self.identity = {
            **({} if extra_identity is None else extra_identity),
            "format": "torch-aot-function-v1",
            "component": component,
            "model": model_identity,
            "source_sha256": [file_fingerprint(path) for path in [*sources, Path(__file__)]],
            "runtime": self.runtime,
        }
        variable = f"VIDMAP_{namespace.upper()}_CACHE_DIR"
        root = (
            Path(os.environ[variable]).expanduser()
            if variable in os.environ
            else Path.home() / ".cache/vidmap" / namespace
        )
        self.root = root / digest(self.identity)
        if net is not None:
            self.train(net.training)

    def prepare_module(self, net):
        return net

    def load_module(self, path):
        return torch.load(path, map_location=f"cuda:{self.runtime['device']}", weights_only=False)

    def _ensure_module(self):
        if self.net is not None:
            return
        if not self.persistent:
            self.net = self.prepare_module(self._factory())
            return
        with graph_entry(self.root, "module") as (directory, build):
            if build:
                self.net = self.prepare_module(self._factory())
                torch.save(self.net, directory / "module.pt")
                _manifest(directory, self.identity, "module.pt", write=True)
            else:
                _manifest(directory, self.identity, "module.pt")
                self.net = self.prepare_module(self.load_module(directory / "module.pt"))

    def _apply(self, fn, recurse=True):
        self.graphs.clear()
        if self._factory is not None and self.net is not None:
            self.net.cpu()
            self.net = None
        return super()._apply(fn, recurse=recurse)

    def forward(self, *args, **kwargs):
        if self.training or not torch.is_inference_mode_enabled():
            raise ValueError("Saved graphs require eval and inference mode")
        if torch.get_float32_matmul_precision() != "highest" or runtime_identity() != self.runtime:
            raise ValueError("Compiler, device or precision settings changed")
        return self._run(args, kwargs)

    def _run(self, args, kwargs):
        if not self.persistent:
            self._ensure_module()
            if "jit" not in self.graphs:
                self.graphs["jit"] = torch.compile(type(self.net).forward, fullgraph=True, dynamic=False)
            return self.graphs["jit"](self.net, *args, **kwargs)
        leaves, structure = tree_flatten((args, kwargs))
        assert all(
            isinstance(value, torch.Tensor) or type(value) in (str, int, float, bool, type(None)) for value in leaves
        )
        tensors = [value for value in leaves if isinstance(value, torch.Tensor)]
        assert tensors
        identity = {
            **self.identity,
            "structure": treespec_dumps(structure),
            "inputs": [tensor_signature(value) if isinstance(value, torch.Tensor) else value for value in leaves],
            "aliases": [
                [
                    i
                    for i, candidate in enumerate(tensors)
                    if value is candidate or value.data_ptr() == candidate.data_ptr()
                ]
                for value in tensors
            ],
            "autocast": torch.is_autocast_enabled("cuda"),
            "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        }
        key = digest(identity)
        if key not in self.graphs:
            self._ensure_module()
            with graph_entry(self.root, key) as (directory, build):
                if build:
                    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

                    # Seeded AOT nonce keys can collide. Isolate their lookup, retaining Triton tuning.
                    with (
                        torch._dynamo.convert_frame.compile_lock,
                        TemporaryDirectory(dir=directory) as temporary,
                        patch.object(AOTAutogradCache, "_get_tmp_dir", return_value=temporary),
                        torch._functorch.config.patch(enable_remote_autograd_cache=False),
                    ):
                        graph = torch.compile(type(self.net).forward, fullgraph=True, dynamic=False).aot_compile(
                            ((self.net, *args), kwargs)
                        )
                    graph.save_compiled_function(str(directory / "compiled.pt"))
                    _manifest(directory, identity, "compiled.pt", write=True)
                else:
                    _manifest(directory, identity, "compiled.pt")
                    with (directory / "compiled.pt").open("rb") as stream:
                        graph = torch.compiler.load_compiled_function(
                            stream, f_globals=type(self.net).forward.__globals__
                        )
                # Validate execution before publishing a newly built program.
                result = graph(self.net, *args, **kwargs)
            self.graphs[key] = graph
            return result
        return self.graphs[key](self.net, *args, **kwargs)

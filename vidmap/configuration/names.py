"""Canonical stage-config roots, names, and storage slugs."""

from pathlib import Path, PurePosixPath

CONFIG_DIR = Path(__file__).parents[1] / "configs"
FRONTEND_CONFIG_DIR = CONFIG_DIR / "frontend"
MAPPING_CONFIG_DIR = CONFIG_DIR / "mapping"

DEFAULT_FRONTEND_CONFIG = "uncalib/base"
DEFAULT_MAPPING_CONFIG = "uncalib/base"
FRONTEND_CONFIG_FILENAME = "frontend_config.yaml"
MAPPING_CONFIG_FILENAME = "mapping_config.yaml"


def resolve_config_path(config_name: str, config_root: Path) -> Path:
    """Resolve an extension-free config name without allowing escape from config_root."""
    name = str(config_name)
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or path.suffix in {".yaml", ".yml"}
    ):
        raise ValueError(f"Expected an extension-free config name below the stage config root, got {config_name!r}")
    root = Path(config_root).resolve()
    resolved = (root / f"{name}.yaml").resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Config name escapes its stage config root: {config_name!r}")
    return resolved


def config_name_to_output_slug(config_name: str) -> str:
    """Convert a root-relative config name to its stable output-directory slug."""
    name = str(config_name).replace("\\", "/")
    if name.startswith("configs/"):
        name = name[len("configs/") :]
    for suffix in (".yaml", ".yml"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a root-relative reconstruction config name, got {config_name!r}")
    return "-".join(path.parts)


def benchmark_config_pair_slug(frontend_name: str, mapping_name: str) -> str:
    """Return the stable benchmark directory name for one explicit config pair."""

    mapping_component = str(mapping_name)
    if not mapping_component or Path(mapping_component).name != mapping_component or mapping_component in {".", ".."}:
        raise ValueError(f"Expected one safe resolved mapping name, got {mapping_name!r}")
    return f"{config_name_to_output_slug(frontend_name)}_{mapping_component}"

"""Serialize a frozen config dataclass to YAML on disk."""

import dataclasses
import os
import tempfile
from pathlib import Path

import yaml

from vidmap.configuration.names import FRONTEND_CONFIG_FILENAME, MAPPING_CONFIG_FILENAME


def config_to_dict(obj):
    from vidmap.configuration.config import FrontendRunSpec, MappingRunSpec

    if isinstance(obj, FrontendRunSpec):
        return {
            **config_to_dict(obj.selection),
            "name": obj.name,
            "colmap_runtime": obj.colmap_runtime,
            **config_to_dict(obj.run),
            **config_to_dict(obj.pipeline),
        }
    if isinstance(obj, MappingRunSpec):
        return {
            **config_to_dict(obj.selection),
            "name": obj.name,
            "colmap_runtime": obj.colmap_runtime,
            **config_to_dict(obj.run),
            **config_to_dict(obj.evaluation),
            **config_to_dict(obj.pipeline),
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        values = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            values[field.name] = config_to_dict(value)
        return values
    if isinstance(obj, dict):
        return {k: config_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # Coerce tuples to lists so ``yaml.safe_load`` can roundtrip the output —
        # Pydantic's before-validator accepts list for tuple-typed fields.
        return [config_to_dict(x) for x in obj]
    return obj


def summarize_cfg(config) -> str:
    """Return a resolved, copyable YAML summary of a configuration."""
    from omegaconf import OmegaConf

    source = config if OmegaConf.is_config(config) else OmegaConf.create(config_to_dict(config))
    return OmegaConf.to_yaml(source, resolve=True, sort_keys=False)


def config_file_matches(conf, path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    return yaml.safe_load(path.read_text()) == config_to_dict(conf)


def validate_config_provenance(conf, path: Path, *, overwrite=False, context="Run") -> None:
    path = Path(path)
    if not overwrite and path.exists() and not config_file_matches(conf, path):
        raise ValueError(f"{context} at {path.parent} has different config provenance; use --overwrite to replace it")


def validate_mapping_config_provenance(
    frontend_conf,
    mapping_conf,
    directory: Path,
    *,
    overwrite=False,
    context="Mapping run",
) -> None:
    """Validate the two resolved configs owned by a mapping output directory."""

    directory = Path(directory)
    frontend_path = frontend_config_path(directory)
    mapping_path = directory / MAPPING_CONFIG_FILENAME
    validate_config_provenance(
        frontend_conf,
        frontend_path,
        overwrite=overwrite,
        context=context,
    )
    validate_config_provenance(
        mapping_conf,
        mapping_path,
        overwrite=overwrite,
        context=context,
    )


def mapping_config_files_match(frontend_conf, mapping_conf, directory: Path) -> bool:
    """Report whether a directory records this exact frontend/mapping pair."""

    directory = Path(directory)
    return config_file_matches(
        frontend_conf,
        frontend_config_path(directory),
    ) and config_file_matches(mapping_conf, directory / MAPPING_CONFIG_FILENAME)


def dump_mapping_configs(frontend_conf, mapping_conf, directory: Path, *, overwrite=False) -> None:
    """Atomically write both resolved configs owned by a mapping run."""

    directory = Path(directory)
    frontend_path = directory / FRONTEND_CONFIG_FILENAME
    mapping_path = directory / MAPPING_CONFIG_FILENAME
    if overwrite or not frontend_path.exists():
        dump_config(frontend_conf, frontend_path)
    if overwrite or not mapping_path.exists():
        dump_config(mapping_conf, mapping_path)


def mapping_config_path(directory: Path) -> Path:
    """Return the mapping provenance path."""
    return Path(directory) / MAPPING_CONFIG_FILENAME


def frontend_config_path(directory: Path) -> Path:
    """Return the frontend provenance path."""
    return Path(directory) / FRONTEND_CONFIG_FILENAME


def _replacement_mode(path: Path) -> int:
    if path.exists():
        return path.stat().st_mode & 0o7777
    current_umask = os.umask(0)
    os.umask(current_umask)
    return 0o666 & ~current_umask


def dump_config(conf, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _replacement_mode(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
            temporary = Path(f.name)
            yaml.dump(config_to_dict(conf), f, default_flow_style=False, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

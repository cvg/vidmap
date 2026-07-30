from __future__ import annotations

import inspect
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

import yaml

MANIFEST_NAME = ".vidmap_pycolmap.yaml"


class PycolmapEnvError(RuntimeError):
    pass


def _abs_no_resolve(path: Path, *, repo_root: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return Path(os.path.abspath(os.fspath(path)))


def _load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if data is None:
        raise PycolmapEnvError(f"{path} is empty")
    if not isinstance(data, dict):
        raise PycolmapEnvError(f"{path} must contain a YAML mapping")
    return data


def _git_output(args: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _require_equal(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise PycolmapEnvError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _require_under(label: str, actual: Path, roots: list[Path]) -> None:
    for root in roots:
        if actual == root or root in actual.parents:
            return
    roots_text = ", ".join(str(root) for root in roots)
    raise PycolmapEnvError(f"{label} path mismatch: expected under one of [{roots_text}], got {actual}")


def _active_site_package_roots() -> list[Path]:
    roots = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value is not None:
            root = Path(value).resolve()
            if root not in roots:
                roots.append(root)
    return roots


def _verify_manifest(manifest: dict[str, Any], *, repo_root: Path, manifest_path: Path) -> None:
    expected_python = manifest.get("python")
    if expected_python is not None:
        expected_python_path = _abs_no_resolve(Path(str(expected_python)), repo_root=repo_root)
        actual_python_path = _abs_no_resolve(Path(sys.executable), repo_root=repo_root)
        _require_equal("python executable", str(actual_python_path), str(expected_python_path))

    pycolmap_source_value = manifest.get("pycolmap_source")
    if not pycolmap_source_value:
        raise PycolmapEnvError(f"{manifest_path} must define pycolmap_source")
    pycolmap_source = _abs_no_resolve(Path(str(pycolmap_source_value)), repo_root=repo_root)
    if not pycolmap_source.exists():
        raise PycolmapEnvError(f"pycolmap_source does not exist: {pycolmap_source}")
    if not (pycolmap_source / ".git").exists():
        raise PycolmapEnvError(f"pycolmap_source is not a git checkout: {pycolmap_source}")

    import pycolmap

    expected_source_root = pycolmap_source.resolve()
    expected_python_dir = (pycolmap_source / "python").resolve()
    module_file = Path(pycolmap.__file__).resolve()
    _require_under("pycolmap import", module_file, [expected_python_dir])

    core_file = Path(inspect.getfile(pycolmap.Reconstruction)).resolve()
    _require_under("pycolmap _core", core_file, [expected_source_root, *_active_site_package_roots()])

    expected_version = manifest.get("expected_version")
    if expected_version is not None:
        _require_equal(
            "pycolmap version",
            str(getattr(pycolmap, "__version__", None)),
            str(expected_version),
        )

    expected_ref = manifest.get("pycolmap_ref")
    if expected_ref is not None:
        actual_ref = _git_output(["rev-parse", "HEAD"], cwd=pycolmap_source)
        _require_equal("pycolmap git ref", actual_ref, str(expected_ref))

    expected_branch = manifest.get("pycolmap_branch")
    if expected_branch is not None:
        actual_branch = _git_output(["branch", "--show-current"], cwd=pycolmap_source)
        _require_equal("pycolmap git branch", actual_branch, str(expected_branch))


def assert_declared_pycolmap_environment(repo_root: Path) -> None:
    """Fail if this worktree declares a pycolmap pairing and the runtime differs."""

    manifest_path = repo_root / MANIFEST_NAME
    if not manifest_path.exists():
        return
    manifest = _load_manifest(manifest_path)
    _verify_manifest(manifest, repo_root=repo_root, manifest_path=manifest_path)
    print(f"PYCOLMAP_ENV_VERIFIED {manifest_path}", flush=True)

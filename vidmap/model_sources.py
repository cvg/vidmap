"""Resolve model packages from a source checkout or installed wheel."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


def model_package_root(import_name: str, checkout_package: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    source_root = project_root / checkout_package
    if source_root.is_dir():
        return source_root.resolve()
    return (project_root / import_name).resolve()


def _module_locations(module) -> tuple[Path, ...]:
    locations = [Path(path).resolve() for path in getattr(module, "__path__", ())]
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        locations.append(Path(module_file).resolve())
    return tuple(locations)


def import_model_package(import_name: str, package_root: Path):
    """Import a model package from one explicit package directory."""
    expected = package_root.resolve()
    if not expected.is_dir():
        raise RuntimeError(f"Packaged {import_name} source is unavailable: {expected}")

    loaded = sys.modules.get(import_name)
    if loaded is None:
        init = expected / "__init__.py"
        if init.is_file():
            spec = importlib.util.spec_from_file_location(
                import_name,
                init,
                submodule_search_locations=[str(expected)],
            )
        else:
            spec = importlib.machinery.ModuleSpec(import_name, loader=None, is_package=True)
            spec.submodule_search_locations = [str(expected)]
        if spec is None:
            raise RuntimeError(f"Cannot load {import_name} from {expected}")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[import_name] = loaded
        try:
            if spec.loader is not None:
                spec.loader.exec_module(loaded)
        except BaseException:
            sys.modules.pop(import_name, None)
            raise

    locations = _module_locations(loaded)
    if not locations or any(not location.is_relative_to(expected) for location in locations):
        raise RuntimeError(f"{import_name} was loaded outside the pinned source at {expected}")
    return loaded

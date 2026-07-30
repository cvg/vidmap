"""Public frontend API and local-media command."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vidmap.frontend.cli import build_parser, config_from_args, main
    from vidmap.frontend.pipeline import Frontend

__all__ = ["Frontend", "build_parser", "config_from_args", "main"]


def __getattr__(name: str) -> Any:
    if name == "Frontend":
        from .pipeline import Frontend

        value = Frontend
    elif name in {"build_parser", "config_from_args", "main"}:
        from . import cli

        value = getattr(cli, name)
    else:
        raise AttributeError(f"module 'vidmap.frontend' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

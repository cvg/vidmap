"""Mapper package."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vidmap.mapper.mapper import Mapper

__all__ = ["Mapper"]


def __getattr__(name: str) -> Any:
    if name == "Mapper":
        from vidmap.mapper.mapper import Mapper

        globals()[name] = Mapper
        return Mapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

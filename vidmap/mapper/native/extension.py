"""Lazy access to the installed native SfM extension."""

from __future__ import annotations

from functools import cached_property
from types import ModuleType

from vidmap.mapper.runtime import load_mapping_runtime


class _NativeMappingExtension:
    @cached_property
    def module(self) -> ModuleType:
        return load_mapping_runtime()

    def __getattr__(self, name: str):
        return getattr(self.module, name)


native = _NativeMappingExtension()

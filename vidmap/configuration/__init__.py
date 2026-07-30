"""Public surface of the typed-config module.

The two stage models are exposed lazily so importing this package does not
eagerly load both dependency trees.
"""

from typing import TYPE_CHECKING, Any

from vidmap.configuration.dump import dump_config, summarize_cfg

if TYPE_CHECKING:
    from vidmap.configuration.config import FrontendConfig, FrontendRunSpec, MappingConfig, MappingRunSpec

__all__ = [
    "FrontendConfig",
    "FrontendRunSpec",
    "MappingConfig",
    "MappingRunSpec",
    "dump_config",
    "summarize_cfg",
]

_CONFIG_EXPORTS = frozenset({"FrontendConfig", "FrontendRunSpec", "MappingConfig", "MappingRunSpec"})


def __getattr__(name: str) -> Any:
    if name in _CONFIG_EXPORTS:
        from vidmap.configuration import config

        value = getattr(config, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'vidmap.configuration' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

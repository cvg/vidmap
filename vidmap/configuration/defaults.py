"""Defaults for run selection, storage, and evaluation outside pipeline configs."""

from dataclasses import dataclass, field
from typing import Optional

from vidmap.mapper.options import ReplayCacheOptions

DEFAULT_WATE_WINDOWS_M = (10, 25, 50, 100)
DEFAULT_WATE_AUC_PERCENT = 5.0
DEFAULT_WATE_AUC_THRESHOLDS_M = tuple(window * DEFAULT_WATE_AUC_PERCENT / 100.0 for window in DEFAULT_WATE_WINDOWS_M)
DEFAULT_WATE_AUC_FULL_THRESHOLDS = (DEFAULT_WATE_AUC_PERCENT / 100.0,)


@dataclass(frozen=True)
class TargetSelectionOptions:
    scene: Optional[list[str]] = None
    mode: str = "minimal"
    testset_id: Optional[list[str]] = None


@dataclass(frozen=True)
class FrontendRunOptions:
    frontend_cache_root: Optional[str] = None
    deterministic_frontend: bool = False
    pre_geom_db_stop: bool = False
    pre_geom_repro_dir: Optional[str] = None
    replay_cache: ReplayCacheOptions = field(default_factory=ReplayCacheOptions)


@dataclass(frozen=True)
class MappingRunOptions:
    workspace_outputs: bool = False
    output_root: Optional[str] = None
    frontend_cache_root: Optional[str] = None
    persist_intermediate_reconstructions: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.persist_intermediate_reconstructions, bool):
            raise TypeError(
                "persist_intermediate_reconstructions must be bool, "
                f"got {type(self.persist_intermediate_reconstructions).__name__}"
            )


@dataclass(frozen=True)
class EvaluationOptions:
    windowed_ate_sizes: list[int] = field(default_factory=lambda: list(DEFAULT_WATE_WINDOWS_M))
    windowed_ate_auc_thresholds: list[float] = field(default_factory=lambda: list(DEFAULT_WATE_AUC_THRESHOLDS_M))
    windowed_ate_auc_full_thresholds: list[float] = field(
        default_factory=lambda: list(DEFAULT_WATE_AUC_FULL_THRESHOLDS)
    )
    windowed_ate_auc_missing_error: Optional[float] = None
    eval_on_full_gt_timeline: bool = False

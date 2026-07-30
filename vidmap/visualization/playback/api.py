from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Literal

from .outputs import finalize_output_file, temporary_sibling
from .scene import DEPTH_LIFT_RADIUS_UI_POINTS

Resolution = tuple[int, int]
DEFAULT_RESOLUTION: Resolution = (1280, 720)
Mode = Literal["playback", "final", "flythrough"]
View = Literal["topdown", "side", "isometric", "follow"]
Theme = Literal["dark", "light", "neon"]
Reconstruction = Literal["final", "ba", "gp"] | Path
SparsePointMode = Literal["progressive", "static"]


@dataclass(frozen=True)
class Playback:
    """A GP or GP+BA playback specification."""

    source: str | Path
    _: KW_ONLY
    gp_only: bool = False
    mode: Mode = "playback"
    view: View | None = None
    theme: Theme = "dark"
    reconstruction: str | Path | None = None
    align_to_ground_truth: bool = True
    ground_truth: str | Path | None = None
    image_dir: str | Path | None = None
    sparse_point_mode: SparsePointMode | None = None
    sparse_point_covariance_percentile: float | None = None
    depth_lift: bool = False
    depth_lift_keep: bool = False
    depth_lift_refit_scale: bool | None = None
    depth_maps: str | Path | None = None
    depth_lift_stride: int = 1
    depth_lift_max: float = 20.0
    depth_lift_point_radius: float = DEPTH_LIFT_RADIUS_UI_POINTS
    duration_fraction: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _path(self.source, "source"))
        if not isinstance(self.gp_only, bool):
            raise TypeError("gp_only must be a bool")
        if self.mode not in {"playback", "final", "flythrough"}:
            raise ValueError("mode must be 'playback', 'final', or 'flythrough'")
        view = self.view or ("follow" if self.mode == "flythrough" else "topdown")
        if view not in {"topdown", "side", "isometric", "follow"}:
            raise ValueError("view must be 'topdown', 'side', 'isometric', or 'follow'")
        if view == "follow" and self.mode != "flythrough":
            raise ValueError("view='follow' is only valid with mode='flythrough'")
        if self.gp_only and self.mode == "flythrough":
            raise ValueError("gp_only is not valid with mode='flythrough'")
        if self.mode == "flythrough":
            if self.reconstruction is None:
                object.__setattr__(self, "reconstruction", "final")
            reconstruction = self.reconstruction
            if isinstance(reconstruction, str) and reconstruction not in {
                "final",
                "ba",
                "gp",
            }:
                reconstruction = _path(reconstruction, "reconstruction")
            elif not isinstance(reconstruction, (str, Path)):
                raise TypeError("reconstruction must be 'final', 'ba', 'gp', or a path")
            object.__setattr__(self, "reconstruction", reconstruction)
        elif self.reconstruction is not None:
            raise ValueError("reconstruction is only valid with mode='flythrough'")
        if self.image_dir is not None:
            if self.mode != "flythrough":
                raise ValueError("image_dir is only valid with mode='flythrough'")
            object.__setattr__(self, "image_dir", _path(self.image_dir, "image_dir"))
        sparse_point_mode = self.sparse_point_mode
        if sparse_point_mode is None:
            sparse_point_mode = "static" if self.mode == "flythrough" else "progressive"
            object.__setattr__(self, "sparse_point_mode", sparse_point_mode)
        if sparse_point_mode not in {"progressive", "static"}:
            raise ValueError("sparse_point_mode must be 'progressive' or 'static'")
        if sparse_point_mode != "progressive" and self.mode != "flythrough":
            raise ValueError("sparse_point_mode is only configurable with mode='flythrough'")
        percentile = self.sparse_point_covariance_percentile
        if percentile is None and self.mode == "flythrough":
            percentile = 90.0
            object.__setattr__(self, "sparse_point_covariance_percentile", percentile)
        if percentile is not None:
            if not isinstance(percentile, (int, float)) or isinstance(percentile, bool) or not 0 < percentile <= 100:
                raise ValueError("sparse_point_covariance_percentile must be in (0, 100]")
            if self.mode != "flythrough":
                raise ValueError("sparse_point_covariance_percentile is only valid with mode='flythrough'")
            object.__setattr__(self, "sparse_point_covariance_percentile", float(percentile))
        if not isinstance(self.depth_lift, bool):
            raise TypeError("depth_lift must be a bool")
        if not isinstance(self.depth_lift_keep, bool):
            raise TypeError("depth_lift_keep must be a bool")
        refit_scale_requested = self.depth_lift_refit_scale is True
        if self.depth_lift_refit_scale is None:
            object.__setattr__(self, "depth_lift_refit_scale", True)
        elif not isinstance(self.depth_lift_refit_scale, bool):
            raise TypeError("depth_lift_refit_scale must be a bool")
        if self.depth_lift_keep or refit_scale_requested:
            object.__setattr__(self, "depth_lift", True)
        if self.depth_lift and self.mode != "flythrough":
            raise ValueError("depth_lift is only valid with mode='flythrough'")
        if self.depth_maps is not None:
            if self.mode != "flythrough":
                raise ValueError("depth_maps is only valid with mode='flythrough'")
            object.__setattr__(self, "depth_maps", _path(self.depth_maps, "depth_maps"))
        if (
            not isinstance(self.depth_lift_stride, int)
            or isinstance(self.depth_lift_stride, bool)
            or self.depth_lift_stride < 1
        ):
            raise ValueError("depth_lift_stride must be a positive integer")
        if (
            not isinstance(self.depth_lift_max, (int, float))
            or isinstance(self.depth_lift_max, bool)
            or self.depth_lift_max <= 0
        ):
            raise ValueError("depth_lift_max must be positive")
        if (
            not isinstance(self.depth_lift_point_radius, (int, float))
            or isinstance(self.depth_lift_point_radius, bool)
            or self.depth_lift_point_radius <= 0
        ):
            raise ValueError("depth_lift_point_radius must be positive")
        if (
            not isinstance(self.duration_fraction, (int, float))
            or isinstance(self.duration_fraction, bool)
            or not 0 < self.duration_fraction <= 1
        ):
            raise ValueError("duration_fraction must be in (0, 1]")
        if self.duration_fraction != 1.0 and self.mode != "flythrough":
            raise ValueError("duration_fraction is only valid with mode='flythrough'")
        object.__setattr__(self, "duration_fraction", float(self.duration_fraction))
        if self.theme not in {"dark", "light", "neon"}:
            raise ValueError("theme must be 'dark', 'light', or 'neon'")
        if not isinstance(self.align_to_ground_truth, bool):
            raise TypeError("align_to_ground_truth must be a bool")
        if self.ground_truth is not None:
            if not self.align_to_ground_truth:
                raise ValueError("ground_truth cannot be combined with align_to_ground_truth=False")
            object.__setattr__(self, "ground_truth", _path(self.ground_truth, "ground_truth"))
        object.__setattr__(self, "view", view)

    def save_rrd(
        self,
        output: str | Path,
        *,
        resolution: Resolution = DEFAULT_RESOLUTION,
        overwrite: bool = False,
    ) -> Path:
        """Write this playback to a Rerun recording."""
        from . import rrd

        output_path = Path(output)
        _check_output(output_path, ".rrd", overwrite=overwrite)
        resolved_resolution = _resolution(resolution)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = temporary_sibling(output_path, ".tmp.rrd")
        try:
            if self.mode == "flythrough":
                from .flythrough import build_model_flythrough_sequence

                sequence = build_model_flythrough_sequence(self, resolved_resolution)
            else:
                from .solver_sequence import build_solver_playback_sequence

                sequence = build_solver_playback_sequence(self, resolved_resolution)
            rrd.write(sequence, temporary)
            finalize_output_file(temporary, output_path, overwrite=overwrite)
            return output_path
        finally:
            temporary.unlink(missing_ok=True)


def _resolution(value: Resolution) -> Resolution:
    if not isinstance(value, tuple) or len(value) != 2 or not all(isinstance(item, int) for item in value):
        raise TypeError("resolution must be a (width, height) tuple of integers")
    if value[0] <= 0 or value[1] <= 0:
        raise ValueError("resolution dimensions must be positive")
    return value


def _path(value: str | Path, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise TypeError(f"{label} must be a non-empty path")
    return Path(value)


def _check_output(path: Path, suffix: str, *, overwrite: bool) -> None:
    if path.suffix.lower() != suffix:
        raise ValueError(f"Playback output must end in {suffix}: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path} (pass overwrite=True to replace it)")

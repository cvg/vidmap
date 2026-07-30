"""Refine the 3D points of an existing VidMap reconstruction with explicit opt-in."""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import yaml

from vidmap.mapper.inputs import MapperInputs
from vidmap.mapper.inputs.database import remove_database_sidecars
from vidmap.mapper.inputs.loader import MappingProblemLoader
from vidmap.mapper.options import ReplayCacheOptions
from vidmap.mapper.options.mapper import SetupOptions
from vidmap.mapper.options.refinement import BAOptions
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.mapper.runtime import load_mapping_runtime
from vidmap.mapper.stages.bundle_adjustment.adjuster import BundleAdjuster
from vidmap.utils.logging import configure_logging

logger = logging.getLogger("vidmap.refine_points")


@dataclass(frozen=True)
class PointRefinementConfig:
    """The saved mapping values needed by point-only refinement."""

    setup: SetupOptions
    ba: BAOptions
    depth_stddev_multiplier: float


def _saved_mapping_config(run_dir: Path) -> Path:
    path = run_dir / "mapping_config.yaml"
    if path.is_file():
        return path
    raise FileNotFoundError(f"No saved mapping config found in {run_dir}")


def load_point_refinement_config(run_dir: str | Path) -> PointRefinementConfig:
    """Load only the stable mapping fields consumed by point refinement."""
    path = _saved_mapping_config(Path(run_dir))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping configuration")
    mapper = raw["mapper"]
    setup = SetupOptions(**mapper["setup"])
    ba = BAOptions(**mapper["ba"])
    depth_stddev_multiplier = float(mapper["mdrp"]["depth_stddev_multiplier"])
    if depth_stddev_multiplier <= 0:
        raise ValueError(f"{path}: mdrp.depth_stddev_multiplier must be positive")
    return PointRefinementConfig(
        setup=setup,
        ba=ba,
        depth_stddev_multiplier=depth_stddev_multiplier,
    )


def _camera_poses(reconstruction: pycolmap.Reconstruction) -> dict[int, np.ndarray]:
    return {
        int(image_id): np.column_stack(
            (
                np.asarray(reconstruction.image(image_id).cam_from_world().rotation.matrix(), dtype=np.float64),
                np.asarray(reconstruction.image(image_id).cam_from_world().translation, dtype=np.float64),
            )
        )
        for image_id in reconstruction.reg_image_ids()
    }


def _camera_parameters(reconstruction: pycolmap.Reconstruction) -> dict[int, np.ndarray]:
    return {
        int(camera_id): np.asarray(camera.params, dtype=np.float64).copy()
        for camera_id, camera in reconstruction.cameras.items()
    }


def _validate_fixed_cameras(
    before_poses: dict[int, np.ndarray],
    before_parameters: dict[int, np.ndarray],
    reconstruction: pycolmap.Reconstruction,
) -> tuple[float, float]:
    after_poses = _camera_poses(reconstruction)
    after_parameters = _camera_parameters(reconstruction)
    if before_poses.keys() != after_poses.keys():
        raise RuntimeError("Point refinement changed the registered image set")
    if before_parameters.keys() != after_parameters.keys():
        raise RuntimeError("Point refinement changed the camera set")
    pose_delta = max(
        (float(np.max(np.abs(before_poses[key] - after_poses[key]))) for key in before_poses),
        default=0.0,
    )
    parameter_delta = max(
        (float(np.max(np.abs(before_parameters[key] - after_parameters[key]))) for key in before_parameters),
        default=0.0,
    )
    if pose_delta > 1e-12 or parameter_delta > 1e-12:
        raise RuntimeError(
            "Point refinement changed fixed camera state: "
            f"max pose delta={pose_delta:.3g}, max intrinsic delta={parameter_delta:.3g}"
        )
    return pose_delta, parameter_delta


def _replace_reconstruction(source: Path, staging: Path) -> None:
    """Replace a reconstruction only after its staged successor is complete."""
    backup_root = Path(tempfile.mkdtemp(prefix=f".{source.name}.", suffix=".backup", dir=source.parent))
    backup = backup_root / source.name
    try:
        source.rename(backup)
    except BaseException:
        shutil.rmtree(backup_root, ignore_errors=True)
        raise
    try:
        staging.rename(source)
    except BaseException:
        backup.rename(source)
        shutil.rmtree(backup_root, ignore_errors=True)
        raise
    shutil.rmtree(backup_root, ignore_errors=True)


def refine_run_points(
    run_dir: str | Path,
    *,
    in_place: bool = False,
    mapper_inputs: str | Path | None = None,
    reconstruction: str = "rec",
) -> Path:
    """Retriangulate and optimize 3D points in a completed run in place."""
    if not in_place:
        raise ValueError("Point refinement requires explicit in_place=True because it replaces the reconstruction")
    load_mapping_runtime()
    run_dir = Path(run_dir).expanduser().resolve()
    reconstruction_path = Path(reconstruction)
    if (
        not reconstruction
        or reconstruction_path.is_absolute()
        or reconstruction_path.name != reconstruction
        or reconstruction in {".", ".."}
    ):
        raise ValueError(f"Reconstruction must be one directory name inside the run, got {reconstruction!r}")
    source = run_dir / reconstruction
    if source.is_symlink():
        raise ValueError(f"Refusing to replace a symlinked reconstruction: {source}")
    mapper_inputs_dir = run_dir / "mapper_inputs" if mapper_inputs is None else Path(mapper_inputs).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Source reconstruction is missing: {source}")

    config = load_point_refinement_config(run_dir)
    inputs = MapperInputs.from_directory(mapper_inputs_dir)
    staging = Path(tempfile.mkdtemp(prefix=f".{source.name}.point-refinement.", dir=source.parent))
    working_database = staging / "database_complete.db"
    try:
        stage_inputs = MappingProblemLoader(
            options=config.setup,
            use_geocalib=inputs.boundary_option("use_geocalib"),
            inputs=inputs,
            sfm_outputs_dir=staging,
            replay=ReplayCache(ReplayCacheOptions(), staging),
        ).load()
        checkpoint = pycolmap.Reconstruction(source)
        before_poses = _camera_poses(checkpoint)
        before_parameters = _camera_parameters(checkpoint)
        points_before = checkpoint.num_points3D()
        observations_before = checkpoint.compute_num_observations()

        stage_inputs.solve_state.import_checkpoint(checkpoint)
        adjuster = BundleAdjuster(
            solve_state=stage_inputs.solve_state,
            options=config.ba,
            depth_stddev_multiplier=config.depth_stddev_multiplier,
            focal_uncertainty=stage_inputs.focal_uncertainty,
            output_dir=staging,
            replay=ReplayCache(ReplayCacheOptions(), staging),
        )
        adjuster.prepare_workspace()
        if not adjuster.run_post_annealing_point_refinement():
            raise RuntimeError("Point-only refinement solve failed; original reconstruction was not changed")
        pose_delta, parameter_delta = _validate_fixed_cameras(
            before_poses,
            before_parameters,
            adjuster.reconstruction,
        )

        remove_database_sidecars(working_database)
        working_database.unlink(missing_ok=True)
        adjuster.reconstruction.write(staging)
        validated = pycolmap.Reconstruction(staging)
        _validate_fixed_cameras(before_poses, before_parameters, validated)
        _replace_reconstruction(source, staging)
        logger.info(
            "Point-only refinement replaced %s: points %d -> %d, observations %d -> %d, "
            "max camera-pose delta %.3g, max intrinsic delta %.3g",
            source,
            points_before,
            validated.num_points3D(),
            observations_before,
            validated.compute_num_observations(),
            pose_delta,
            parameter_delta,
        )
        return source
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Retriangulate and refine only 3D points in an existing VidMap run.",
    )
    parser.add_argument("run", help="Completed run containing rec/, mapper_inputs/, and its saved config.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        required=True,
        help="Replace the selected reconstruction after successful point-only refinement.",
    )
    parser.add_argument("--mapper-inputs", help="Override RUN/mapper_inputs.")
    parser.add_argument("--reconstruction", default="rec", help="Source reconstruction directory inside RUN.")
    parser.add_argument("-v", "--verbose", type=int, default=1, metavar="LEVEL")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    refine_run_points(
        args.run,
        in_place=args.in_place,
        mapper_inputs=args.mapper_inputs,
        reconstruction=args.reconstruction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Frontend-only execution over selected benchmark targets."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from vidmap import depth_artifacts
from vidmap.benchmark.results import FrontendResultStore
from vidmap.frontend.identity import FrontendIdentity
from vidmap.frontend.pipeline import Frontend
from vidmap.frontend.preparation.cameras import prepare_capture, resolve_camera_preparation
from vidmap.mapper.inputs import MapperInputs
from vidmap.utils.device import resolve_device
from vidmap.utils.logging import log_context

if TYPE_CHECKING:
    import torch

    from vidmap.datasets.capture import PreparedCapture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrontendTargetResult:
    """One finalized frontend boundary produced or reused for a target."""

    identity: FrontendIdentity
    mapper_inputs: MapperInputs
    reused: bool

    @property
    def tag(self) -> str:
        return self.identity.tag

    @property
    def path(self):
        return self.mapper_inputs.directory


@dataclass(frozen=True)
class FrontendRunResult:
    """Structured result for a finite frontend selection."""

    targets: tuple[FrontendTargetResult, ...]
    failure_count: int


def run_local_frontend(
    conf,
    input_path: str | Path | PreparedCapture,
    *,
    workspace: str | Path,
    imnames=None,
    force_frontend: bool = False,
    cache_depth_maps: bool = False,
    mapper_inputs_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> FrontendTargetResult:
    """Process local media or a prepared capture into a finalized boundary."""
    from vidmap.datasets.capture import PreparedCapture, prepare_scene_capture
    from vidmap.datasets.local import LocalImageParser

    workspace = Path(workspace).expanduser()
    input_snapshot = None
    preparation_plan = None
    if isinstance(input_path, PreparedCapture):
        if imnames is not None:
            raise ValueError("Capture inputs already supply calibration and complete frames")
        scene_parser = input_path
    elif conf.pipeline.rig is not None:
        if imnames is not None:
            raise ValueError("Rig configuration supplies calibration and complete frames")
        rig = conf.pipeline.rig
        from vidmap.datasets.prepare.rig_capture import load_capture, prepare_rig_capture

        source_capture = load_capture(input_path, rig)
        preparation_plan = resolve_camera_preparation(source_capture, conf.pipeline)
        input_snapshot = prepare_rig_capture(
            source_capture,
            workspace / "prepared_capture",
            camera_ids_to_undistort=preparation_plan.undistort_camera_ids,
        )
        scene_parser = input_snapshot.capture
    else:
        from vidmap.reconstruction import prepare_reconstruction_images

        if conf.pipeline.camera_priors.initialization != "predicted":
            raise ValueError("Calibrated local input requires a rig configuration, including for one camera")
        image_dir = prepare_reconstruction_images(input_path, workspace)
        scene_parser = LocalImageParser(image_dir=image_dir, imnames=imnames)
    from vidmap.reconstruction import write_local_input_provenance

    scene_parser = prepare_scene_capture(scene_parser)
    if preparation_plan is None:
        preparation_plan = resolve_camera_preparation(scene_parser, conf.pipeline)
        if preparation_plan.undistort_camera_ids:
            from vidmap.datasets.prepare.rig_capture import prepare_rig_capture

            input_snapshot = prepare_rig_capture(
                scene_parser,
                workspace / "prepared_capture",
                camera_ids_to_undistort=preparation_plan.undistort_camera_ids,
            )
            scene_parser = input_snapshot.capture
    if input_snapshot is None and (scene_parser.kind is not None or scene_parser.frame_timestamps_ns is not None):
        input_snapshot = scene_parser.snapshot()
    device = resolve_device(device)
    image_dir = scene_parser.rgb_dir
    reference_image_ids = tuple(scene_parser.rec.images)
    identity = FrontendIdentity.from_config(
        conf,
        dataset="local",
        scene="local" if scene_parser.kind is None else scene_parser.scene,
        mode="all",
        testset_id="input",
        reference_image_ids=reference_image_ids,
        source_fingerprint=None if input_snapshot is None else input_snapshot.fingerprint(),
    )
    temporary_dir = workspace / "tmp_cache_dir"
    mapper_inputs_dir = (
        workspace / "mapper_inputs" if mapper_inputs_path is None else Path(mapper_inputs_path).expanduser()
    )
    full_depth_maps_path = workspace / "full_depth_maps.h5"
    if mapper_inputs_dir.exists() and not force_frontend:
        mapper_inputs = MapperInputs.from_directory(
            mapper_inputs_dir,
            expected_identity=identity.as_dict(),
        )
        mapper_inputs.validate()
        write_local_input_provenance(
            mapper_inputs.directory,
            image_dir,
            overwrite=True,
        )
        if cache_depth_maps:
            import h5py

            from vidmap.frontend.depth import cache_full_depth_maps_posthoc
            from vidmap.frontend.image_dataset import FrameSequence

            selected = set()
            with h5py.File(mapper_inputs.depth_maps_path, "r") as hfile:

                def collect_depth_group(name, item):
                    if isinstance(item, h5py.Group) and {"depth", "valid"}.issubset(item):
                        selected.add(str(name))

                hfile.visititems(collect_depth_group)
            selected = frozenset(selected)
            frames = FrameSequence.from_scene(scene_parser, rig_options=conf.pipeline.rig)
            image_names = tuple(name for name in frames.names if name in selected)
            if len(image_names) != len(selected):
                missing = sorted(selected - set(image_names))
                raise ValueError(f"Mapper-input depth names are absent from local media: {missing}")
            cache_full_depth_maps_posthoc(
                scene_parser=prepare_capture(scene_parser, preparation_plan),
                image_names=image_names,
                sampled_depth_path=mapper_inputs.depth_maps_path,
                output_path=full_depth_maps_path,
                options=conf.pipeline.depth,
                device=device,
            )
            depth_artifacts.write_reference_paths(
                workspace,
                full_depth_maps_path,
                mapper_inputs.depth_maps_path,
            )
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        return FrontendTargetResult(identity, mapper_inputs, reused=True)
    if mapper_inputs_path is not None:
        raise FileNotFoundError(f"Finalized mapper inputs not found: {mapper_inputs_dir}")
    if input_snapshot is None:
        input_snapshot = scene_parser.snapshot()
        identity = replace(identity, source_fingerprint=input_snapshot.fingerprint())
    frontend = Frontend(
        conf=conf.pipeline,
        replay_cache=conf.run.replay_cache,
        sample_name="all",
        deterministic_frontend=conf.run.deterministic_frontend,
        pre_geom_repro_dir=conf.run.pre_geom_repro_dir,
        cache_dir=temporary_dir / "cache",
        sfm_outputs_dir=temporary_dir / "work",
        input_snapshot=input_snapshot,
        preparation_plan=preparation_plan,
        reference_image_names=scene_parser.imnames,
        force_recompute=force_frontend,
        cache_full_depth_maps=cache_depth_maps,
        frontend_tag=identity.tag,
        namespace_cache_by_config=False,
        mapper_inputs_dir=mapper_inputs_dir,
        device=device,
    )
    mapper_inputs = frontend.run(frontend_identity=identity.as_dict())
    mapper_inputs.validate()
    write_local_input_provenance(
        mapper_inputs.directory,
        image_dir,
        overwrite=True,
    )
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    return FrontendTargetResult(identity, mapper_inputs, reused=False)


class FrontendRunner:
    def __init__(self, conf, selection, run_options=None, *, output_root=None) -> None:
        self.conf = conf
        self.selection = selection
        self.run_options = run_options
        self.output_root = output_root

    def run_with_results(self, *, force_frontend: bool = False) -> FrontendRunResult:
        """Run frontend and return every concrete finalized boundary."""
        from vidmap.datasets.capture import prepare_scene_capture
        from vidmap.reconstruction import write_local_input_provenance

        failure_count = 0
        completed = []
        for (
            target,
            scene_parser,
            reference_image_names,
        ) in self.selection.prepared_targets():
            results = FrontendResultStore(
                self.conf,
                self.selection.dataset_layout,
                scene=target.scene,
                testset_type=target.testset_type,
                testset_desc=target.testset_desc,
                output_root=self.output_root,
            )
            with log_context(
                dataset=self.selection.dataset_spec.name,
                scene=target.scene,
                mode=target.testset_type,
                testset=target.testset_desc,
                config=self.conf.name,
            ):
                try:
                    scene_parser = prepare_scene_capture(scene_parser)
                    input_snapshot = (
                        scene_parser.snapshot()
                        if scene_parser.kind is not None or scene_parser.frame_timestamps_ns is not None
                        else None
                    )
                    identity = FrontendIdentity.from_config(
                        self.conf,
                        dataset=self.selection.dataset_spec.name,
                        scene=target.scene,
                        mode=target.testset_type,
                        testset_id=target.testset_desc,
                        reference_image_ids=target.ref_imids,
                        source_fingerprint=None if input_snapshot is None else input_snapshot.fingerprint(),
                    )
                    existing = (
                        None
                        if force_frontend
                        else results.reusable_mapper_inputs(self.selection.dataset_layout, identity)
                    )
                    if existing is not None:
                        existing.validate()
                        write_local_input_provenance(
                            existing.directory,
                            scene_parser.rgb_dir,
                            overwrite=True,
                        )
                        result = FrontendTargetResult(identity, existing, reused=True)
                        completed.append(result)
                        logger.info(
                            "Finalized mapper inputs: tag=%s path=%s (reused)",
                            result.tag,
                            result.path,
                        )
                        continue
                    if input_snapshot is None:
                        input_snapshot = scene_parser.snapshot()
                        identity = replace(identity, source_fingerprint=input_snapshot.fingerprint())
                    results.cache_dir.mkdir(parents=True, exist_ok=True)
                    frontend = Frontend(
                        conf=self.conf.pipeline,
                        replay_cache=self.conf.run.replay_cache,
                        sample_name=target.testset_type,
                        deterministic_frontend=self.conf.run.deterministic_frontend,
                        pre_geom_repro_dir=self.conf.run.pre_geom_repro_dir,
                        cache_dir=results.cache_dir,
                        sfm_outputs_dir=results.resolved_frontend_work_dir(),
                        input_snapshot=input_snapshot,
                        reference_image_names=reference_image_names,
                        force_recompute=force_frontend,
                        frontend_tag=identity.tag,
                        namespace_cache_by_config=True,
                        mapper_inputs_dir=results.resolved_mapper_inputs_dir(),
                        device=getattr(self.run_options, "device", None),
                    )
                    if self.conf.run.pre_geom_db_stop:
                        frontend.run_pre_geom()
                    else:
                        mapper_inputs = frontend.run(frontend_identity=identity.as_dict())
                        write_local_input_provenance(
                            mapper_inputs.directory,
                            scene_parser.rgb_dir,
                            overwrite=True,
                        )
                        result = FrontendTargetResult(identity, mapper_inputs, reused=False)
                        completed.append(result)
                        logger.info(
                            "Finalized mapper inputs: tag=%s path=%s",
                            result.tag,
                            result.path,
                        )
                except Exception:
                    if getattr(self.run_options, "terminate_on_error", False):
                        raise
                    failure_count += 1
                    logger.exception("Frontend target failed")
        return FrontendRunResult(tuple(completed), failure_count)

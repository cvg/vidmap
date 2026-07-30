"""Benchmark workflow orchestration."""

import logging
from dataclasses import replace

from vidmap.benchmark.results import BenchmarkResultStore
from vidmap.benchmark.trajectory import (
    AbsoluteTrajectoryEvaluator,
    AbsoluteTrajectoryOptions,
    align_reconstruction_for_evaluation,
)
from vidmap.frontend.identity import FrontendIdentity
from vidmap.mapper.inputs import MapperInputs
from vidmap.reconstruction import run_mapping
from vidmap.run_options import RunOptions
from vidmap.utils.logging import log_context
from vidmap.utils.trajectory import compute_ate_statistics, reconstruction_to_evo_trajectory, remap_poses_to_timeline

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Map and evaluate every selected prepared target."""

    def __init__(
        self,
        conf,
        frontend_conf,
        selection,
        run_options: RunOptions | None = None,
        *,
        output_name: str | None = None,
    ) -> None:
        self.conf = conf
        self.frontend_conf = frontend_conf
        self.selection = selection
        self.run_options = RunOptions() if run_options is None else run_options
        self.output_name = output_name
        self.evaluator = AbsoluteTrajectoryEvaluator(options=AbsoluteTrajectoryOptions.from_config(conf.evaluation))

    def run(
        self,
        mapper_inputs_dir=None,
        *,
        overwrite_results: bool = False,
    ) -> int:
        """Run every selected target and return the number that failed."""
        failure_count = 0
        for (
            target,
            scene_parser,
            reference_image_names,
        ) in self.selection.prepared_targets():
            with log_context(
                dataset=self.selection.dataset_spec.name,
                scene=target.scene,
                mode=target.testset_type,
                testset=target.testset_desc,
                config=self.conf.name,
            ):
                try:
                    self.run_case(
                        target,
                        scene_parser,
                        reference_image_names,
                        mapper_inputs_dir=mapper_inputs_dir,
                        overwrite_results=overwrite_results,
                    )
                except Exception:
                    if self.run_options.terminate_on_error:
                        raise
                    failure_count += 1
                    logger.exception("Benchmark target failed")
        return failure_count

    def run_case(
        self,
        target,
        scene_parser,
        reference_image_names,
        *,
        mapper_inputs_dir=None,
        overwrite_results: bool = False,
    ):
        """Run one prepared target through an explicit validated boundary."""
        output_kwargs = {"output_name": self.output_name} if self.output_name is not None else {}
        results = BenchmarkResultStore(
            self.conf,
            self.selection.dataset_layout,
            frontend_conf=self.frontend_conf,
            scene=target.scene,
            testset_type=target.testset_type,
            testset_desc=target.testset_desc,
            **output_kwargs,
        )
        logger.info("Output directory: %s", results.output_dir)
        frontend_identity = FrontendIdentity.from_config(
            self.frontend_conf,
            dataset=self.selection.dataset_spec.name,
            scene=target.scene,
            mode=target.testset_type,
            testset_id=target.testset_desc,
            reference_image_ids=target.ref_imids,
        )
        if isinstance(mapper_inputs_dir, MapperInputs):
            reusable_mapper_inputs = replace(
                mapper_inputs_dir,
                expected_identity=frontend_identity.as_dict(),
            )
        elif mapper_inputs_dir is not None:
            reusable_mapper_inputs = MapperInputs.from_directory(
                mapper_inputs_dir,
                expected_identity=frontend_identity.as_dict(),
            )
        else:
            reusable_mapper_inputs = results.require_mapper_inputs(frontend_identity)
        if not results.prepare(
            frontend_only=False,
            overwrite_results=overwrite_results,
            save_3d_html=self.run_options.save_3d_html,
            save_playback_trace=self.run_options.save_playback_trace,
            playback_trace_stride=self.run_options.playback_trace_stride,
            playback_trace_point_cap=self.run_options.playback_trace_point_cap,
        ):
            return None
        return BenchmarkCaseRunner(
            self.conf,
            self.frontend_conf,
            self.run_options,
            results,
            evaluator=self.evaluator,
        ).run(
            scene_parser,
            reference_image_names,
            gt_evaluable=self.selection.scene_has_public_ground_truth(target.scene),
            mapper_inputs_dir=reusable_mapper_inputs,
            overwrite_results=overwrite_results,
        )


class BenchmarkCaseRunner:
    """Run mapping, alignment, and evaluation for one selected case."""

    def __init__(self, conf, frontend_conf, run_options: RunOptions, results, *, evaluator=None) -> None:
        self.conf = conf
        self.frontend_conf = frontend_conf
        self.run_options = run_options
        self.results = results
        self.evaluator = evaluator

    def reconstruct(
        self,
        scene_parser,
        *,
        mapper_inputs_dir,
        overwrite_results: bool = False,
    ):
        return run_mapping(
            self.conf,
            frontend_conf=self.frontend_conf,
            mapper_inputs=mapper_inputs_dir,
            run_options=self.run_options,
            scene_parser=scene_parser,
            output_dir=self.results.output_dir,
            scene_name=self.results.scene,
            overwrite_outputs=overwrite_results,
        )

    def run(
        self,
        scene_parser,
        reference_image_names,
        *,
        gt_evaluable: bool,
        mapper_inputs_dir,
        overwrite_results: bool = False,
    ):
        estimated_reconstruction = self.reconstruct(
            scene_parser,
            mapper_inputs_dir=mapper_inputs_dir,
            overwrite_results=overwrite_results,
        )
        if estimated_reconstruction is None:
            return None
        if estimated_reconstruction.num_reg_images() == 0:
            logger.warning("No registered images; skipping evaluation")
            if gt_evaluable:
                self.results.write_zero_registration()
            else:
                self.results.write_reconstruction(estimated_reconstruction)
            return estimated_reconstruction

        if not gt_evaluable:
            logger.info("Public ground truth is unavailable; saving reconstruction without trajectory metrics")
            self.results.write_reconstruction(estimated_reconstruction)
            return estimated_reconstruction

        if self.evaluator is None:
            raise RuntimeError("Benchmark evaluation requires an evaluator")
        align_reconstruction_for_evaluation(estimated_reconstruction, scene_parser.rec)
        estimated_trajectory = reconstruction_to_evo_trajectory(estimated_reconstruction)
        aligned_ground_truth_reconstruction = remap_poses_to_timeline(
            estimated_reconstruction, scene_parser.rec
        ).reconstruction
        ground_truth_trajectory = reconstruction_to_evo_trajectory(aligned_ground_truth_reconstruction)
        absolute_trajectory_error = compute_ate_statistics(ground_truth_trajectory, estimated_trajectory)
        self.results.write_ate(absolute_trajectory_error)
        trajectory_evaluation = self.evaluator.evaluate(
            estimated_reconstruction=estimated_reconstruction,
            ground_truth_reconstruction=scene_parser.rec,
        )
        trajectory_metrics = trajectory_evaluation.summarize(verbose=logger.isEnabledFor(logging.INFO))
        self.results.write_metrics(absolute_trajectory_error, trajectory_metrics)
        self.results.write_reconstruction(estimated_reconstruction)
        self.results.write_trajectory(estimated_trajectory, ground_truth_trajectory)

        return estimated_reconstruction

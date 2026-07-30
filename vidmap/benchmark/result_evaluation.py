"""Evaluate saved benchmark reconstructions."""

import logging
from collections import defaultdict

import matplotlib
import pycolmap
from tqdm import tqdm

from vidmap.benchmark.trajectory import AbsoluteTrajectoryEvaluator, AbsoluteTrajectoryOptions
from vidmap.datasets.selection import PreparedTargetSelection
from vidmap.run_options import RunOptions
from vidmap.utils.logging import log_context, progress_bars_enabled

from .results import BenchmarkResultStore

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class BenchmarkResultEvaluator:
    """Evaluate saved reconstructions selected from prepared dataset targets."""

    def __init__(
        self,
        conf,
        frontend_conf_or_dataset,
        dataset=None,
        run_options: RunOptions | None = None,
        *,
        output_name: str | None = None,
    ) -> None:
        self.conf = conf
        if dataset is None:
            dataset = frontend_conf_or_dataset
            self.frontend_conf = None
        else:
            self.frontend_conf = frontend_conf_or_dataset
        self.output_name = output_name
        self.run_options = RunOptions() if run_options is None else run_options
        self.selection = PreparedTargetSelection(conf.selection, dataset)
        self.evaluator = AbsoluteTrajectoryEvaluator(options=AbsoluteTrajectoryOptions.from_config(conf.evaluation))

    @staticmethod
    def remap_image_ids(reconstruction, ground_truth):
        """Return a reconstruction whose image IDs match ground truth names."""
        ground_truth_ids = {image.name: image_id for image_id, image in ground_truth.images.items()}
        id_mapping = {
            image_id: ground_truth_ids[image.name]
            for image_id, image in reconstruction.images.items()
            if image.name in ground_truth_ids and image_id != ground_truth_ids[image.name]
        }
        if not id_mapping:
            return reconstruction

        remapped = pycolmap.Reconstruction()
        for camera in reconstruction.cameras.values():
            remapped.add_camera_with_trivial_rig(camera)
        for image_id, image in reconstruction.images.items():
            remapped_image = pycolmap.Image(
                image_id=id_mapping.get(image_id, image_id),
                camera_id=image.camera_id,
                name=image.name,
            )
            if image.has_pose:
                remapped.add_image_with_trivial_frame(remapped_image, image.cam_from_world())
            else:
                remapped.add_image_with_trivial_frame(remapped_image)
        for point3d in reconstruction.points3D.values():
            remapped.add_point3D(point3d)
        return remapped

    def evaluate_case(self, case, scene_parser, outputs, recalls, wates):
        """Load, evaluate, and aggregate one persisted benchmark reconstruction."""
        try:
            results = BenchmarkResultStore(
                self.conf,
                self.selection.dataset_layout,
                frontend_conf=self.frontend_conf,
                scene=case.scene,
                testset_type=case.testset_type,
                testset_desc=case.testset_desc,
                output_name=self.output_name,
            )
            reconstruction = pycolmap.Reconstruction(results.output_dir / "rec")
            reconstruction = self.remap_image_ids(reconstruction, scene_parser.rec)
            evaluation = self.evaluator.evaluate(
                estimated_reconstruction=reconstruction,
                ground_truth_reconstruction=scene_parser.rec,
            )
            logger.info(
                "Evaluating scene %s, testset %s/%s",
                case.scene,
                case.testset_type,
                case.testset_desc,
            )

            summary = evaluation.summarize(verbose=logger.isEnabledFor(logging.INFO))
            outputs[case.scene][case.testset_type].append(summary["all"])
            if "recall" in summary:
                recalls[case.scene][case.testset_type].append(summary["recall"])
            for key in (key for key in summary if key.startswith("wate_") or key.startswith("wrre_")):
                target = wates[case.scene][key][case.testset_type]
                if key.startswith("wate_auc_errors_"):
                    target.extend(summary[key])
                else:
                    target.append(summary[key])
            return True, case.scene
        except Exception as error:
            if self.run_options.terminate_on_error:
                raise error
            logger.exception("Error processing %s: %s", case.scene, error)
            return False, case.scene

    def evaluate_saved(self):
        """Evaluate every selected reconstruction saved on disk."""
        scenes = self.selection.dataset_layout.scenes
        outputs = {scene: defaultdict(list) for scene in scenes}
        recalls = {scene: defaultdict(list) for scene in scenes}
        wates = {scene: defaultdict(lambda: defaultdict(list)) for scene in scenes}
        counts = {scene: {"completed": 0, "failed": 0} for scene in scenes}
        for case, scene_parser, _ in tqdm(
            self.selection.prepared_targets(),
            disable=not progress_bars_enabled(),
        ):
            with log_context(
                dataset=self.selection.dataset_spec.name,
                scene=case.scene,
                mode=case.testset_type,
                testset=case.testset_desc,
                config=self.conf.name,
            ):
                ok, scene = self.evaluate_case(case, scene_parser, outputs, recalls, wates)
                counts[scene]["completed" if ok else "failed"] += 1
        return outputs, counts, recalls, wates

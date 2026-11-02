"""Evaluate saved benchmark reconstructions."""

import logging
from collections import defaultdict

import matplotlib
import pycolmap
import yaml
from tqdm import tqdm

from vidmap.benchmark.trajectory import AbsoluteTrajectoryEvaluator
from vidmap.configuration.dump import frontend_config_path
from vidmap.datasets.selection import PreparedTargetSelection
from vidmap.frontend.options.rig import RigOptions
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
        rig = None if self.frontend_conf is None else self.frontend_conf.pipeline.rig
        self.evaluator = AbsoluteTrajectoryEvaluator.from_config(conf, rig)

    def _report_group(self, case):
        # EuRoC reports individual recordings, independently of their storage scene.
        if self.selection.dataset_spec.name == "euroc":
            return f"{case.scene}-{case.recording}"
        return case.scene

    def evaluate_case(self, case, scene_parser, outputs, recalls, wates):
        """Load, evaluate, and aggregate one persisted benchmark reconstruction."""
        group = self._report_group(case)
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
            evaluator = self.evaluator
            frontend_path = frontend_config_path(results.output_dir)
            if frontend_path.is_file():
                frontend = yaml.safe_load(frontend_path.read_text())
                rig = frontend["rig"] if "rig" in frontend else None
                evaluator = AbsoluteTrajectoryEvaluator.from_config(
                    self.conf, None if rig is None else RigOptions(**rig)
                )
            evaluation = evaluator.evaluate(
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
            outputs[group][case.testset_type].append(summary["all"])
            if "recall" in summary:
                recalls[group][case.testset_type].append(summary["recall"])
            for key in (key for key in summary if key.startswith("wate_") or key.startswith("wrre_")):
                target = wates[group][key][case.testset_type]
                if key.startswith("wate_auc_errors_"):
                    target.extend(summary[key])
                else:
                    target.append(summary[key])
            return True, group
        except Exception as error:
            if self.run_options.terminate_on_error:
                raise error
            logger.exception("Error processing %s: %s", case.scene, error)
            return False, group

    def evaluate_saved(self):
        """Evaluate every selected reconstruction saved on disk."""
        groups = self.selection.dataset_layout.scenes
        if self.selection.dataset_spec.name == "euroc":
            groups = tuple(
                self._report_group(case) for _, cases in self.selection.iter_scene_targets() for case in cases
            )
        outputs = {group: defaultdict(list) for group in groups}
        recalls = {group: defaultdict(list) for group in groups}
        wates = {group: defaultdict(lambda: defaultdict(list)) for group in groups}
        counts = {group: {"completed": 0, "failed": 0} for group in groups}
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
                ok, group = self.evaluate_case(case, scene_parser, outputs, recalls, wates)
                counts[group]["completed" if ok else "failed"] += 1
        return outputs, counts, recalls, wates

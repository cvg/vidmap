"""Benchmark output paths and result persistence."""

import logging
from pathlib import Path

import numpy as np
import yaml

from vidmap.benchmark.trajectory import convert_numpy_scalars_to_builtin, windowed_ate_auc_fields
from vidmap.benchmark.trajectory_artifacts import write_trajectory_artifacts
from vidmap.configuration.config import MappingRunSpec
from vidmap.configuration.dump import mapping_config_files_match, validate_mapping_config_provenance
from vidmap.configuration.names import benchmark_config_pair_slug, config_name_to_output_slug
from vidmap.frontend.identity import FrontendIdentity
from vidmap.mapper.inputs import MANIFEST_NAME, MapperInputs, validate_mapper_inputs_identity

logger = logging.getLogger(__name__)


def validate_playback_trace_artifact(
    path: Path,
    *,
    iteration_stride: int,
    point_cap: int | None,
) -> None:
    """Require one complete playback trace with the requested sampling."""
    from vidmap.mapper.playback_trace_storage import PlaybackTrace

    path = Path(path)
    trace = PlaybackTrace.load(path)
    expected = {
        "sampling": {
            "iteration_stride": iteration_stride,
            "point_cap": point_cap,
        }
    }
    if trace.metadata != expected:
        raise ValueError(
            f"{path}: playback trace sampling metadata does not match the requested settings; "
            f"expected {expected!r}, found {trace.metadata!r}"
        )


class BenchmarkResultStore:
    """Own persisted outputs for one scene and testset."""

    sfm_outputs_dir_template = "{exp_dir}/reconstruction/{testset_type}/{scene}/{testset_desc}/{conf}"
    cache_output_dir_template = "{h5_dir}/{scene}/{traj}"

    def __init__(
        self,
        conf,
        dataset,
        *,
        frontend_conf,
        scene: str,
        testset_type: str,
        testset_desc: str,
        output_name: str | None = None,
    ) -> None:
        self.conf = conf
        self.frontend_conf = frontend_conf
        self.scene = scene
        self.testset_type = testset_type
        self.testset_desc = testset_desc
        self.output_dir = self.resolved_output_dir(
            conf,
            dataset,
            frontend_conf=frontend_conf,
            scene=scene,
            testset_type=testset_type,
            testset_desc=testset_desc,
            output_name=output_name,
        )
        self.cache_dir = self.resolved_cache_dir(conf, dataset, scene=scene, testset_desc=testset_desc)

    @classmethod
    def resolved_output_dir(
        cls,
        conf,
        dataset,
        *,
        frontend_conf,
        scene: str,
        testset_type: str,
        testset_desc: str,
        output_name: str | None = None,
    ) -> Path:
        if conf.run.output_root is not None:
            exp_dir = Path(conf.run.output_root).expanduser()
        elif conf.run.workspace_outputs:
            exp_dir = Path.cwd() / ".benchmark_outputs"
        else:
            exp_dir = dataset.default_exp_dir
        if not isinstance(conf, MappingRunSpec):
            raise TypeError("Benchmark reconstruction outputs require a mapping config")
        if output_name is None:
            conf_name = benchmark_config_pair_slug(frontend_conf.name, conf.name)
        else:
            conf_name = str(output_name)
            if not conf_name or Path(conf_name).name != conf_name or "\\" in conf_name or conf_name in {".", ".."}:
                raise ValueError(f"Expected one safe benchmark output name, got {output_name!r}")
        return Path(
            cls.sfm_outputs_dir_template.format(
                exp_dir=exp_dir,
                testset_type=testset_type,
                scene=scene,
                testset_desc=testset_desc,
                conf=conf_name,
            )
        )

    @classmethod
    def resolved_cache_dir(cls, conf, dataset, *, scene: str, testset_desc: str) -> Path:
        if conf.run.frontend_cache_root is not None:
            cache_root = Path(conf.run.frontend_cache_root).expanduser()
            return cache_root / dataset.name / scene / testset_desc
        return Path(
            cls.cache_output_dir_template.format(
                h5_dir=dataset.default_cache_dir,
                scene=scene,
                traj=testset_desc,
            )
        )

    @staticmethod
    def _resolved_frontend_slug(conf) -> str:
        slug = config_name_to_output_slug(conf.name)
        if not slug or slug in {".", ".."} or Path(slug).name != slug:
            raise ValueError(f"Benchmark frontend requires one safe config slug, got {slug!r}")
        return slug

    def resolved_frontend_dir(self) -> Path:
        """Return the tagged frontend namespace independent of mapping outputs."""

        return self.cache_dir / self._resolved_frontend_slug(self.frontend_conf) / self.testset_type

    def resolved_mapper_inputs_dir(self) -> Path:
        return self.resolved_frontend_dir() / "mapper_inputs"

    def resolved_frontend_work_dir(self) -> Path:
        return self.resolved_frontend_dir() / "work"

    def reusable_mapper_inputs_candidate_exists(self, dataset=None) -> bool:
        return (self.resolved_mapper_inputs_dir() / MANIFEST_NAME).is_file()

    def reusable_mapper_inputs(self, dataset, identity: FrontendIdentity) -> MapperInputs | None:
        """Resolve the compatible finalized boundary for the selected frontend tag."""
        expected_identity = identity.as_dict()
        tagged = self.resolved_mapper_inputs_dir()
        if (tagged / MANIFEST_NAME).is_file():
            validate_mapper_inputs_identity(tagged, expected_identity)
            return MapperInputs.from_directory(tagged, expected_identity=expected_identity)
        return None

    def require_mapper_inputs(self, identity: FrontendIdentity) -> MapperInputs:
        mapper_inputs = self.reusable_mapper_inputs(None, identity)
        if mapper_inputs is None:
            path = self.resolved_mapper_inputs_dir()
            raise FileNotFoundError(
                f"Finalized mapper inputs are missing for frontend {identity.tag!r}: {path}. "
                "Run python -m vidmap.run_for_benchmark --frontend-only with the corresponding frontend config first."
            )
        return mapper_inputs

    def experiment_exists(self) -> bool:
        return bool((self.output_dir / "rec" / "images.bin").exists())

    def completed_run_exists(self) -> bool:
        """Validate provenance and report whether this exact run is complete."""
        validate_mapping_config_provenance(
            self.frontend_conf,
            self.conf,
            self.output_dir,
            context="Benchmark run",
        )
        return mapping_config_files_match(self.frontend_conf, self.conf, self.output_dir) and self.experiment_exists()

    def prepare(
        self,
        *,
        frontend_only: bool,
        overwrite_results: bool,
        save_playback_trace: bool = False,
        playback_trace_stride: int = 3,
        playback_trace_point_cap: int | None = None,
    ) -> bool:
        """Validate provenance and report whether this run needs mapping."""
        validate_mapping_config_provenance(
            self.frontend_conf,
            self.conf,
            self.output_dir,
            overwrite=overwrite_results,
            context="Benchmark run",
        )
        completed = (
            not frontend_only
            and mapping_config_files_match(self.frontend_conf, self.conf, self.output_dir)
            and self.experiment_exists()
        )
        if completed and not overwrite_results:
            missing_artifacts = self.missing_requested_artifacts(
                save_playback_trace=save_playback_trace,
                playback_trace_stride=playback_trace_stride,
                playback_trace_point_cap=playback_trace_point_cap,
            )
            if missing_artifacts:
                missing = ", ".join(missing_artifacts)
                raise FileExistsError(
                    f"Benchmark reconstruction already exists at {self.output_dir}, but requested artifacts "
                    f"are missing or invalid: {missing}. Use --overwrite to regenerate the reconstruction "
                    "and requested artifacts."
                )
            logger.info(
                "Skipping %s because results and requested artifacts already exist",
                self.scene,
            )
            return False
        return True

    def missing_requested_artifacts(
        self,
        *,
        save_playback_trace: bool,
        playback_trace_stride: int,
        playback_trace_point_cap: int | None,
    ) -> list[str]:
        missing = []
        if save_playback_trace:
            trace_dir = self.output_dir / "playback_trace"
            try:
                validate_playback_trace_artifact(
                    trace_dir,
                    iteration_stride=playback_trace_stride,
                    point_cap=playback_trace_point_cap,
                )
            except (OSError, ValueError):
                missing.append("playback_trace")
        return missing

    def write_zero_registration(self) -> None:
        with open(self.output_dir / "ate.txt", "w") as file:
            file.write("inf\n")

    def write_ate(self, ate) -> None:
        ate_file = self.output_dir / "ate.txt"
        with open(ate_file, "w") as file:
            file.write(f"{ate}\n")
        logger.info("Saved ATE to %s", ate_file)

    def write_metrics(self, ate, eval_summary) -> None:
        metrics = {"ate": ate}
        wate_lines = []
        wate_auc_lines = []

        if "wate_path_length" in eval_summary:
            metrics["wate_path_length"] = eval_summary["wate_path_length"]
            wate_lines.append(f"path_length_m\t{eval_summary['wate_path_length']:.6f}")

        for key in sorted(eval_summary):
            if key == "wate_path_length":
                continue
            value = eval_summary[key]
            if key.startswith("wate_") and not key.startswith("wate_auc_errors_"):
                metrics[key] = value
                if isinstance(value, (int, float, np.integer, np.floating)):
                    wate_lines.append(f"{key}\t{float(value):.6f}")
                else:
                    wate_lines.append(f"{key}\t{value}")

        for key in sorted(eval_summary):
            if not key.startswith("wate_auc_errors_"):
                continue
            window_s = key.removeprefix("wate_auc_errors_")
            if window_s == "full":
                thresholds = list(self.conf.evaluation.windowed_ate_auc_full_thresholds)
                window_size = "full"
            else:
                thresholds = list(self.conf.evaluation.windowed_ate_auc_thresholds)
                window_size = int(window_s)
            for auc_key, auc in windowed_ate_auc_fields(window_size, eval_summary[key], thresholds).items():
                metrics[auc_key] = auc
                wate_auc_lines.append(f"{auc_key}\t{auc * 100:.6f}")

        with open(self.output_dir / "eval_metrics.tsv", "w") as file:
            file.write("metric\tvalue\n")
            for key in sorted(metrics):
                value = metrics[key]
                if isinstance(value, (int, float, np.integer, np.floating)):
                    file.write(f"{key}\t{float(value):.12g}\n")
                else:
                    file.write(f"{key}\t{value}\n")

        with open(self.output_dir / "eval_summary.yaml", "w") as file:
            yaml.safe_dump(convert_numpy_scalars_to_builtin(eval_summary), file, sort_keys=True)

        if wate_lines:
            with open(self.output_dir / "wate.txt", "w") as file:
                file.write("\n".join(wate_lines) + "\n")
        if wate_auc_lines:
            with open(self.output_dir / "wate_auc.txt", "w") as file:
                if "wate_path_length" in eval_summary:
                    file.write(f"path_length_m\t{eval_summary['wate_path_length']:.6f}\n")
                file.write("\n".join(wate_auc_lines) + "\n")

    def write_trajectory(self, estimated_trajectory, ground_truth_trajectory) -> None:
        write_trajectory_artifacts(self.output_dir, estimated_trajectory, ground_truth_trajectory)
        logger.info("Saved trajectory artifacts to %s", self.output_dir)

    def write_reconstruction(self, reconstruction) -> None:
        recdir = self.output_dir / "rec"
        recdir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving reconstruction to %s", recdir)
        reconstruction.write(recdir)


class FrontendResultStore(BenchmarkResultStore):
    """Frontend-owned paths for one scene/testset target."""

    def __init__(
        self,
        conf,
        dataset,
        *,
        scene: str,
        testset_type: str,
        testset_desc: str,
        output_root=None,
    ) -> None:
        self.conf = conf
        self.frontend_conf = conf
        self.scene = scene
        self.testset_type = testset_type
        self.testset_desc = testset_desc
        self.cache_dir = self.resolved_cache_dir(conf, dataset, scene=scene, testset_desc=testset_desc)
        if output_root is None:
            self.output_dir = super().resolved_frontend_work_dir()
        else:
            self.output_dir = (
                Path(output_root).expanduser()
                / "reconstruction"
                / testset_type
                / scene
                / testset_desc
                / self._resolved_frontend_slug(conf)
            )

    def resolved_frontend_work_dir(self) -> Path:
        return self.output_dir

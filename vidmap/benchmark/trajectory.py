import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field

import numpy as np
import pycolmap
from pydantic import ConfigDict

from vidmap.configuration.validators import dataclass as pydantic_dataclass
from vidmap.utils.trajectory import align_reconstruction_to_reference_sequence, remap_poses_to_timeline

logger = logging.getLogger(__name__)


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class AbsoluteTrajectoryOptions:
    ate_thresholds: list[float] = dc_field(default_factory=lambda: [0.05, 0.1, 0.5, 1, 10, 100])
    gt_scale: float = 1.0
    windowed_ate_sizes: list[int] = dc_field(default_factory=list)
    windowed_ate_auc_thresholds: list[float] = dc_field(default_factory=list)
    windowed_ate_auc_full_thresholds: list[float] = dc_field(default_factory=list)
    windowed_ate_auc_missing_error: float | None = None
    windowed_ate_min_points: int = 3
    eval_on_full_gt_timeline: bool = False

    @classmethod
    def from_config(cls, conf) -> "AbsoluteTrajectoryOptions":
        """Build evaluation options from the benchmark configuration."""
        return cls(
            windowed_ate_sizes=list(conf.windowed_ate_sizes),
            windowed_ate_auc_thresholds=list(conf.windowed_ate_auc_thresholds),
            windowed_ate_auc_full_thresholds=list(conf.windowed_ate_auc_full_thresholds),
            windowed_ate_auc_missing_error=conf.windowed_ate_auc_missing_error,
            eval_on_full_gt_timeline=conf.eval_on_full_gt_timeline,
        )


_EPS = np.finfo(float).eps * 4.0


def error_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for threshold in thresholds:
        last_index = np.searchsorted(errors, threshold)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], threshold]
        aucs.append(np.round(np.trapz(r, x=e) / threshold, 4))
    return aucs


def error_recall(errors, thresholds):
    errors = np.array(errors)
    return [np.round(np.sum(errors <= threshold) / len(errors), 4) for threshold in thresholds]


def _format_wate_auc_threshold(threshold):
    return f"{threshold:g}"


def wate_auc_result_key(window_size, threshold):
    if window_size == "full":
        return f"wate_auc_full@{_format_wate_auc_threshold(threshold * 100.0)}pct"
    return f"wate_auc_{window_size}@{_format_wate_auc_threshold(threshold)}m"


def windowed_error_auc(errors, thresholds):
    finite_errors = [float(error) for error in errors if np.isfinite(error)]
    if not finite_errors or not thresholds:
        return []
    return [float(value) for value in error_auc(finite_errors, thresholds)]


def windowed_ate_auc_fields(window_size, errors, thresholds):
    aucs = windowed_error_auc(errors, thresholds)
    return {wate_auc_result_key(window_size, threshold): auc for threshold, auc in zip(thresholds, aucs, strict=True)}


def align_reconstruction_for_evaluation(reconstruction, ground_truth_reconstruction) -> None:
    """Apply the frozen paper-default two-pass Sim(3) alignment."""
    # Paper-reproduction outputs align once for visualization and refine that
    # result before evaluation. Keep this numerical contract explicit.
    for _ in range(2):
        _, similarity = align_reconstruction_to_reference_sequence(reconstruction, ground_truth_reconstruction)
        reconstruction.transform(similarity)


def logmap_so3(R):
    """Logmap at the identity.
    Returns canonical coordinates of rotation.
    cfo, 2015/08/13

    """
    R11 = R[0, 0]
    R12 = R[0, 1]
    R13 = R[0, 2]
    R21 = R[1, 0]
    R22 = R[1, 1]
    R23 = R[1, 2]
    R31 = R[2, 0]
    R32 = R[2, 1]
    R33 = R[2, 2]
    tr = np.trace(R)
    omega = np.empty((3,), dtype=np.float64)

    # when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, we do something
    # special
    if np.abs(tr + 1.0) < 1e-10:
        if np.abs(R33 + 1.0) > 1e-10:
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R33)) * np.array([R13, R23, 1.0 + R33])
        elif np.abs(R22 + 1.0) > 1e-10:
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R22)) * np.array([R12, 1.0 + R22, R32])
        else:
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R11)) * np.array([1.0 + R11, R21, R31])
    else:
        magnitude = 1.0
        tr_3 = tr - 3.0
        if tr_3 < -1e-7:
            theta = np.arccos((tr - 1.0) / 2.0)
            magnitude = theta / (2.0 * np.sin(theta))
        else:
            # when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            # use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            magnitude = 0.5 - tr_3 * tr_3 / 12.0

        omega = magnitude * np.array([R32 - R23, R13 - R31, R21 - R12])

    return omega


def homogeneous_rotation_matrix_from_quaternion(quaternion):
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def align_umeyama_sim3(p_es, p_gt):
    """Return the scale, rotation, and translation of an Umeyama Sim(3) alignment."""
    mu_es = p_es.mean(0)
    mu_gt = p_gt.mean(0)
    es_zero = p_es - mu_es
    gt_zero = p_gt - mu_gt
    n = len(p_es)
    C = (1.0 / n) * gt_zero.T @ es_zero
    sigma2 = (1.0 / n) * np.sum(es_zero**2)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (1.0 / sigma2) * np.trace(np.diag(D) @ S)
    t = mu_gt - s * R @ mu_es
    return s, R, t


def compute_absolute_pose_errors(p_es_aligned, q_es_aligned, p_gt, q_gt):
    e_trans = np.sqrt(np.sum((p_gt - p_es_aligned) ** 2, 1))

    # orientation error
    e_rot = np.zeros(len(e_trans))
    for i in range(np.shape(p_es_aligned)[0]):
        R_we = homogeneous_rotation_matrix_from_quaternion(q_es_aligned[i, :])
        R_wg = homogeneous_rotation_matrix_from_quaternion(q_gt[i, :])
        e_R = np.dot(R_we, np.linalg.inv(R_wg))
        e_rot[i] = np.rad2deg(np.linalg.norm(logmap_so3(e_R[:3, :3])))

    return e_trans, e_rot


def convert_numpy_scalars_to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: convert_numpy_scalars_to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_scalars_to_builtin(value) for value in obj]
    return obj


@dataclass(frozen=True, kw_only=True)
class AbsoluteTrajectoryResult:
    options: AbsoluteTrajectoryOptions
    data: dict

    def summarize(self, *, verbose=False):
        text, summary = self._build_summary(verbose=verbose)
        if verbose:
            covered = len(self.data["summary"]["have_ids"])
            expected = len(self.data["summary"]["should_have_ids"])
            logger.info(
                "Number of covered GTs: %d/%d=%.2f%%",
                covered,
                expected,
                covered / expected * 100,
            )
            logger.info("%s", text.rstrip())
        return summary

    def _build_summary(self, *, verbose):
        summary_dict = {}
        ate_errs = [self.data["summary"][f"AUC-ate@{th*100}"] for th in self.options.ate_thresholds]
        recall_errs = [self.data["summary"][f"Recall-ate@{th*100}"] for th in self.options.ate_thresholds]
        summary_dict["all"] = ate_errs
        summary_dict["recall"] = recall_errs
        if "ate_median" in self.data["summary"]:
            summary_dict["wate_full"] = self.data["summary"]["ate_median"]
            summary_dict["wate_full_mean"] = self.data["summary"]["ate_mean"]
        if "rre_median" in self.data["summary"]:
            summary_dict["wrre_full"] = self.data["summary"]["rre_median"]
            summary_dict["wrre_full_mean"] = self.data["summary"]["rre_mean"]
        full_results = self.data.get("full_results") or {}
        if "gt_path_length" in self.data["summary"]:
            path_length = float(self.data["summary"]["gt_path_length"])
            summary_dict["wate_path_length"] = path_length
            if (
                (self.options.windowed_ate_auc_full_thresholds or self.options.windowed_ate_auc_thresholds)
                and path_length > 0
                and "ate" in full_results
            ):
                missing_error = self.options.windowed_ate_auc_missing_error
                have_names = set(full_results.get("rre", {}))
                full_errors = []
                for name, error in full_results["ate"].items():
                    if missing_error is not None and name not in have_names:
                        error = missing_error
                    if np.isfinite(error):
                        full_errors.append(float(error) / path_length)
                summary_dict["wate_auc_errors_full"] = full_errors
        expected_names = set(full_results.get("ate", {}))
        for window_size in self.options.windowed_ate_sizes:
            median_key = f"wate{window_size}_median"
            if median_key in self.data["summary"]:
                summary_dict[f"wate_{window_size}"] = self.data["summary"][median_key]
                summary_dict[f"wate_{window_size}_mean"] = self.data["summary"][f"wate{window_size}_mean"]
                summary_dict[f"wate_{window_size}_count"] = self.data["summary"][f"wate{window_size}_count"]
            if self.options.windowed_ate_auc_thresholds:
                full_key = f"windowed_ate_{window_size}"
                if full_key in full_results:
                    wate_errors = list(full_results[full_key].values())
                    missing_error = self.options.windowed_ate_auc_missing_error
                    if missing_error is not None:
                        wate_errors.extend([missing_error] * len(expected_names - set(full_results[full_key])))
                    summary_dict[f"wate_auc_errors_{window_size}"] = wate_errors
            rre_median_key = f"wrre{window_size}_median"
            if rre_median_key in self.data["summary"]:
                summary_dict[f"wrre_{window_size}"] = self.data["summary"][rre_median_key]
                summary_dict[f"wrre_{window_size}_mean"] = self.data["summary"][f"wrre{window_size}_mean"]

        out = ""
        if verbose:
            out = f"ATE thesholds: {self.options.ate_thresholds[:-1]}\n"
            out += "AUC:    " + " | ".join([f"{error*100:.2f}" for error in ate_errs[:-1]]) + "\n"
            out += "Recall: " + " | ".join([f"{error*100:.2f}" for error in recall_errs[:-1]]) + "\n"
            wate_medians = []
            wate_means = []
            wate_counts = []
            wrre_medians = []
            wrre_means = []
            for window_size in self.options.windowed_ate_sizes:
                if f"wate_{window_size}" in summary_dict:
                    wate_medians.append(f"{window_size}m={summary_dict[f'wate_{window_size}']:.4f}")
                    wate_means.append(f"{window_size}m={summary_dict[f'wate_{window_size}_mean']:.4f}")
                    wate_counts.append(f"{window_size}m={int(summary_dict[f'wate_{window_size}_count'])}")
                if f"wrre_{window_size}" in summary_dict:
                    wrre_medians.append(f"{window_size}m={summary_dict[f'wrre_{window_size}']:.2f}°")
                    wrre_means.append(f"{window_size}m={summary_dict[f'wrre_{window_size}_mean']:.2f}°")
            path = (
                f"path={self.data['summary']['gt_path_length']:.1f}m | "
                if "gt_path_length" in self.data["summary"]
                else ""
            )
            if wate_medians:
                median_full = (
                    f"full={self.data['summary']['ate_median']:.4f} | " if "ate_median" in self.data["summary"] else ""
                )
                mean_full = (
                    f"full={self.data['summary']['ate_mean']:.4f} | " if "ate_mean" in self.data["summary"] else ""
                )
                out += "WATE (median): " + path + median_full + " | ".join(wate_medians) + "\n"
                out += "WATE (mean):   " + path + mean_full + " | ".join(wate_means) + "\n"
                out += "WATE (count):  " + path + " | ".join(wate_counts) + "\n"
            if wrre_medians:
                median_full = (
                    f"full={self.data['summary']['rre_median']:.2f}° | "
                    if "rre_median" in self.data["summary"]
                    else ""
                )
                mean_full = (
                    f"full={self.data['summary']['rre_mean']:.2f}° | " if "rre_mean" in self.data["summary"] else ""
                )
                out += "WRRE (median): " + median_full + " | ".join(wrre_medians) + "\n"
                out += "WRRE (mean):   " + mean_full + " | ".join(wrre_means) + "\n"
        return out, summary_dict


class AbsoluteTrajectoryEvaluator:
    def __init__(self, *, options: AbsoluteTrajectoryOptions | None = None):
        if options is None:
            options = AbsoluteTrajectoryOptions()
        elif not isinstance(options, AbsoluteTrajectoryOptions):
            raise TypeError(f"Expected AbsoluteTrajectoryOptions, got {type(options).__name__}")
        self.options = options

    def summarize_pose_errors(self, pose_errors):
        pose_error_summary = {}
        absolute_translation_errors = list(pose_errors["ate"].values())
        pose_error_summary["ate_per_image"] = pose_errors["ate"]
        absolute_translation_error_auc = error_auc(absolute_translation_errors, self.options.ate_thresholds)
        for threshold_index, threshold in enumerate(self.options.ate_thresholds):
            pose_error_summary[f"AUC-ate@{threshold*100}"] = absolute_translation_error_auc[threshold_index]
        absolute_translation_error_recall = error_recall(absolute_translation_errors, self.options.ate_thresholds)
        for threshold_index, threshold in enumerate(self.options.ate_thresholds):
            pose_error_summary[f"Recall-ate@{threshold*100}"] = absolute_translation_error_recall[threshold_index]

        if self.options.windowed_ate_sizes:
            # Global median ATE (registered frames only, for WATE reference)
            registered_translation_errors = [v for v in pose_errors["ate"].values() if v < 1e5]
            if registered_translation_errors:
                pose_error_summary["ate_median"] = float(np.median(registered_translation_errors))
                pose_error_summary["ate_mean"] = float(np.mean(registered_translation_errors))
            # Global rotation error (registered frames only, for WRRE reference)
            if "rre" in pose_errors:
                relative_rotation_errors = list(pose_errors["rre"].values())
                if relative_rotation_errors:
                    pose_error_summary["rre_median"] = float(np.median(relative_rotation_errors))
                    pose_error_summary["rre_mean"] = float(np.mean(relative_rotation_errors))
            if "gt_path_length" in pose_errors:
                pose_error_summary["gt_path_length"] = pose_errors["gt_path_length"]

        for window_size in self.options.windowed_ate_sizes:
            windowed_translation_key = f"windowed_ate_{window_size}"
            if windowed_translation_key in pose_errors:
                windowed_translation_errors = list(pose_errors[windowed_translation_key].values())
                if windowed_translation_errors:
                    pose_error_summary[f"wate{window_size}_median"] = float(np.median(windowed_translation_errors))
                    pose_error_summary[f"wate{window_size}_mean"] = float(np.mean(windowed_translation_errors))
                    pose_error_summary[f"wate{window_size}_count"] = len(windowed_translation_errors)
                    pose_error_summary |= windowed_ate_auc_fields(
                        window_size,
                        windowed_translation_errors,
                        self.options.windowed_ate_auc_thresholds,
                    )
            windowed_rotation_key = f"windowed_rre_{window_size}"
            if windowed_rotation_key in pose_errors:
                windowed_rotation_errors = list(pose_errors[windowed_rotation_key].values())
                if windowed_rotation_errors:
                    pose_error_summary[f"wrre{window_size}_median"] = float(np.median(windowed_rotation_errors))
                    pose_error_summary[f"wrre{window_size}_mean"] = float(np.mean(windowed_rotation_errors))

        return pose_error_summary

    def evaluate_aligned_reconstructions(self, estimated_reconstruction, ground_truth_reconstruction):
        results = {}
        errors = self.absolute_pose_errors(estimated_reconstruction, ground_truth_reconstruction)
        results["full_results"] = errors
        results["results"] = {}
        results["summary"] = defaultdict(dict)

        results["results"] |= self.summarize_pose_errors(errors)

        for metric_name, metric_value in results["results"].items():
            if isinstance(metric_value, dict):
                results["summary"][metric_name] = {
                    key: f"{metric_entry_value:.2f}" for key, metric_entry_value in metric_value.items()
                }
            else:
                results["summary"][metric_name] = metric_value
        results["summary"] |= errors["meta"]
        return results

    def evaluate(
        self,
        *,
        estimated_reconstruction: pycolmap.Reconstruction,
        ground_truth_reconstruction: pycolmap.Reconstruction,
    ) -> AbsoluteTrajectoryResult:
        for image_id in estimated_reconstruction.images:
            if image_id not in ground_truth_reconstruction.images:
                raise ValueError(f"Image {image_id} not in ground truth reconstruction")
        num_images = estimated_reconstruction.num_images()
        num_registered_images = estimated_reconstruction.num_reg_images()
        estimated_reconstruction, _ = align_reconstruction_to_reference_sequence(
            estimated_reconstruction,
            ground_truth_reconstruction,
            max_error=1,
        )
        remap = remap_poses_to_timeline(
            ground_truth_reconstruction,
            estimated_reconstruction,
            prefer_exact_keyframes=True,
        )
        data = {
            "summary": {},
            "results": {},
            "full_results": None,
            "num_images": num_images,
            "num_registered_images": num_registered_images,
            "success": False,
            "conf": asdict(self.options),
        }
        ground_truth = ground_truth_reconstruction
        if remap.target_subset is not None and not self.options.eval_on_full_gt_timeline:
            ground_truth = remap.target_subset
        data |= self.evaluate_aligned_reconstructions(remap.reconstruction, ground_truth)

        data["results"]["scale"] = self.options.gt_scale
        data["success"] = True
        return AbsoluteTrajectoryResult(options=self.options, data=data)

    def absolute_pose_errors(self, estim_rec, gt_rec):
        poses_es_world_full = {
            id: image.cam_from_world().inverse() for id, image in estim_rec.images.items() if image.has_pose
        }
        poses_gt_world_full = {
            id: image.cam_from_world().inverse() for id, image in gt_rec.images.items() if image.has_pose
        }
        have_ids = [id for id in poses_es_world_full if id in poses_gt_world_full]
        should_have_ids = list(poses_gt_world_full)

        poses_es_world = [poses_es_world_full[id] for id in have_ids]
        poses_gt_world = [poses_gt_world_full[id] for id in have_ids]
        imnames = [estim_rec.images[id].name for id in have_ids]

        p_es = np.array([pose.translation for pose in poses_es_world])
        q_es = np.array([pose.rotation.quat for pose in poses_es_world])

        p_gt = np.array([pose.translation for pose in poses_gt_world])
        q_gt = np.array([pose.rotation.quat for pose in poses_gt_world])

        if len(p_es) > 0:
            e_trans, e_rot = compute_absolute_pose_errors(p_es, q_es, p_gt, q_gt)
        else:
            e_trans = [1e6] * len(p_gt)

        out = {"ate": {gt_rec.images[image_id].name: 1e6 for image_id in should_have_ids}}
        out["ate"] |= {name: error * self.options.gt_scale for name, error in zip(imnames, e_trans, strict=True)}
        if len(p_es) > 0:
            out["rre"] = {name: error for name, error in zip(imnames, e_rot, strict=True)}
        out["meta"] = {"have_ids": have_ids, "should_have_ids": should_have_ids}

        # Windowed ATE and full-trajectory WATE-AUC path normalization
        if (self.options.windowed_ate_sizes or self.options.windowed_ate_auc_full_thresholds) and len(
            poses_gt_world_full
        ) > 0:
            # Sort by image name (encodes timestamp) for correct path length.
            # With eval_on_full_gt_timeline, windows are laid out on the full GT
            # path while local alignment uses only registered estimated poses.
            gt_window_ids = should_have_ids if self.options.eval_on_full_gt_timeline else have_ids
            gt_window_ids = sorted(gt_window_ids, key=lambda imid: gt_rec.images[imid].name)
            p_gt_sorted = np.array([poses_gt_world_full[imid].translation for imid in gt_window_ids])
            imnames_sorted = [gt_rec.images[imid].name for imid in gt_window_ids]
            # GT cumulative path length
            diffs = np.linalg.norm(np.diff(p_gt_sorted, axis=0), axis=1)
            cum_len = np.concatenate([[0], np.cumsum(diffs)])
            out["gt_path_length"] = float(cum_len[-1])
            min_pts = self.options.windowed_ate_min_points
            L = cum_len[-1]
            for W in self.options.windowed_ate_sizes:
                if W > L:
                    continue
                wate = {}
                wrre = {}
                half = W / 2.0
                for i, imid in enumerate(gt_window_ids):
                    if imid not in poses_es_world_full:
                        continue
                    ci = cum_len[i]
                    # W-length window clamped to [0, L]
                    lo = max(0.0, min(ci - half, L - W))
                    hi = lo + W
                    mask = (cum_len >= lo) & (cum_len <= hi)
                    window_ids = [gt_window_ids[j] for j, keep in enumerate(mask) if keep]
                    window_have_ids = [wid for wid in window_ids if wid in poses_es_world_full]
                    if len(window_have_ids) < min_pts:
                        continue
                    p_es_w = np.array([poses_es_world_full[wid].translation for wid in window_have_ids])
                    p_gt_w = np.array([poses_gt_world_full[wid].translation for wid in window_have_ids])
                    s, R, t = align_umeyama_sim3(p_es_w, p_gt_w)
                    p_aligned_i = s * R @ poses_es_world_full[imid].translation + t
                    wate[imnames_sorted[i]] = np.linalg.norm(p_gt_sorted[i] - p_aligned_i) * self.options.gt_scale
                    # Relative rotation error: apply Sim(3) rotation to estimated, compare with GT
                    R_es_i = homogeneous_rotation_matrix_from_quaternion(poses_es_world_full[imid].rotation.quat)[
                        :3, :3
                    ]
                    R_gt_i = homogeneous_rotation_matrix_from_quaternion(poses_gt_world_full[imid].rotation.quat)[
                        :3, :3
                    ]
                    R_aligned_i = R @ R_es_i
                    R_err = R_aligned_i @ R_gt_i.T
                    wrre[imnames_sorted[i]] = np.rad2deg(np.linalg.norm(logmap_so3(R_err)))
                out[f"windowed_ate_{W}"] = wate
                out[f"windowed_rre_{W}"] = wrre

        return out

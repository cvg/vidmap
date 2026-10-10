"""Load absolute location priors (rotations and 2D-3D correspondences) and add them to the RA, GP and BA problems."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyceres
import pycolmap
from vidmap_native import bundle_adjustment as ba_costs
from vidmap_native import global_positioning as gp_costs
from vidmap_native import rotation_averaging as ra_costs

from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.location_priors import LocationPriorOptions
from vidmap.mapper.sub_reconstruction import detect_covisibility_components

logger = logging.getLogger(__name__)
_TIMESTAMP = re.compile(r"^(\d+(?:\.\d+)?)(?:-|$)")


def _angular_stds_to_xyz_covar(bearings: np.ndarray, angular_stds: np.ndarray) -> np.ndarray:
    """Propagate diagonal image-plane angular covariance to unit bearings."""
    count = bearings.shape[0]
    image_covariance = np.zeros((count, 3, 3), dtype=np.float64)
    image_covariance[:, 0, 0] = angular_stds[:, 0] ** 2
    image_covariance[:, 1, 1] = angular_stds[:, 1] ** 2

    bearing_z = np.abs(bearings[:, 2])[:, None, None]
    outer_product = np.einsum("ni,nj->nij", bearings, bearings)
    jacobian = bearing_z * (np.eye(3, dtype=np.float64)[None, :, :] - outer_product)
    return np.einsum("nij,njk,nlk->nil", jacobian, image_covariance, jacobian)


def _parse_timestamp_us(name: str) -> int | None:
    match = _TIMESTAMP.match(Path(name).stem)
    if match is None:
        return None
    try:
        return int(round(float(match.group(1)) * 1e6))
    except ValueError:
        return None


@dataclass(frozen=True, kw_only=True)
class LocationAnchorPrior:
    """Location prior of one matched VidMap keyframe."""

    image_id: int
    image_name: str
    confidence: float
    R_cam_from_world: np.ndarray
    cov_rot_cam_from_world: np.ndarray
    prior_point_ids: np.ndarray
    points3D_xyz: np.ndarray
    uv_norm: np.ndarray


@dataclass(frozen=True, kw_only=True)
class LocationPriorSet:
    """Location priors matched to the images of a reconstruction."""

    options: LocationPriorOptions
    anchors: dict[int, LocationAnchorPrior]
    mapped_points_xyz: np.ndarray

    def active_image_ids(self, reconstruction: pycolmap.Reconstruction) -> list[int]:
        """Return sorted image_ids of prior anchors that currently have a valid pose."""
        return [
            image_id
            for image_id in sorted(self.anchors.keys())
            if image_id in reconstruction.images and reconstruction.image(image_id).has_pose
        ]

    def num_active_anchors(self, reconstruction: pycolmap.Reconstruction) -> int:
        return len(self.active_image_ids(reconstruction))

    def add_rotation_priors(self, problem, reconstruction: pycolmap.Reconstruction, *, align: bool):
        """Add absolute rotation priors to a rotation averaging problem (Stage 1).

        With align, the current (e.g. spanning-tree) rotations are first rotated into the prior world frame.
        Returns the storage to keep alive while using the problem and the number of priors.
        """
        if not self.options.enabled or not self.options.use_in_rotation_averaging:
            return None, 0
        image_ids, quaternions, covariances, confidences = [], [], [], []
        for image_id in sorted(self.anchors.keys()):
            if image_id not in reconstruction.images:
                continue
            anchor = self.anchors[image_id]
            rot = pycolmap.Rotation3d(np.asarray(anchor.R_cam_from_world, dtype=np.float64))
            cov = np.asarray(anchor.cov_rot_cam_from_world, dtype=np.float64).copy()
            cov = 0.5 * (cov + cov.T)
            eigvals = np.linalg.eigvalsh(cov)
            if eigvals.min() <= 1e-12:
                cov = cov + (1e-10 - min(0.0, float(eigvals.min()))) * np.eye(3, dtype=np.float64)
            image_ids.append(int(image_id))
            quaternions.append(np.asarray(rot.quat, dtype=np.float64))
            covariances.append(cov)
            confidences.append(float(anchor.confidence))
        storage, count = ra_costs.add_rotation_priors(
            problem,
            reconstruction,
            image_ids,
            quaternions,
            covariances,
            confidences,
            weight=float(self.options.ra_weight),
            cauchy_scale=float(self.options.ra_cauchy_scale),
            ref_sigma_deg=float(self.options.ra_ref_sigma_deg),
            min_sigma_deg=float(self.options.ra_min_sigma_deg),
            align=align,
        )
        logger.info("Using %d location rotation priors in rotation averaging", count)
        return storage, count

    @staticmethod
    def _resect_camera_center_from_rays(
        reconstruction: pycolmap.Reconstruction,
        anchor: LocationAnchorPrior,
    ) -> np.ndarray | None:
        """Compute the camera center in prior world coordinates by intersecting 2D-3D rays with the current R_cw."""
        if len(anchor.prior_point_ids) < 3:
            return None
        image = reconstruction.image(anchor.image_id)
        if not image.has_pose:
            return None
        camera = reconstruction.camera(image.camera_id)
        R_cw = np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float64)

        xy_px = anchor.uv_norm * np.array([float(camera.width), float(camera.height)], dtype=np.float64)
        cam_pts = np.asarray(camera.cam_from_img(xy_px), dtype=np.float64)
        if cam_pts.ndim != 2 or cam_pts.shape[0] < 3:
            return None
        bearings_cam = np.column_stack([cam_pts, np.ones(len(cam_pts), dtype=np.float64)])
        norms = np.linalg.norm(bearings_cam, axis=1, keepdims=True)
        valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-9)
        if int(valid.sum()) < 3:
            return None
        bearings_cam = bearings_cam[valid] / norms[valid]
        pts_world = np.asarray(anchor.points3D_xyz[valid], dtype=np.float64)

        # World ray directions: v_world = R_cw^T @ b_cam
        v_world = bearings_cam @ R_cw
        eye = np.eye(3, dtype=np.float64)[None, :, :]
        proj = eye - v_world[:, :, None] @ v_world[:, None, :]
        A = np.sum(proj, axis=0)
        b = np.sum(proj @ pts_world[:, :, None], axis=0)[:, 0]
        eigvals = np.linalg.eigvalsh(A)
        if float(eigvals.min()) <= 1e-3:
            return None
        center = np.linalg.solve(A, b)
        if not np.all(np.isfinite(center)):
            return None

        # Verify cheirality (majority of 3D points must lie in front of resected center along rays)
        depths = np.sum((pts_world - center[None, :]) * v_world, axis=1)
        if float(np.median(depths)) <= 0.05:
            return None

        # Refine (Cx, Cy, Cz, log_f) with robust Cauchy 2D reprojection error so far landmarks
        # (100-300m) or rough initial focal lengths do not bias the resected center along the optical axis.
        from scipy.optimize import least_squares

        f_init = float(camera.mean_focal_length())
        cx, cy = float(camera.width) / 2.0, float(camera.height) / 2.0
        duv = xy_px[valid] - np.array([cx, cy], dtype=np.float64)[None, :]
        x0 = np.array([center[0], center[1], center[2], np.log(max(f_init, 1.0))], dtype=np.float64)

        def _res_fn(x: np.ndarray) -> np.ndarray:
            c = x[:3]
            f = np.exp(np.clip(x[3], 3.0, 10.0))
            p_cam = (pts_world - c[None, :]) @ R_cw.T
            z = np.maximum(p_cam[:, 2], 0.1)
            pred = f * (p_cam[:, :2] / z[:, None])
            return (pred - duv).ravel() / 2.0

        try:
            sol = least_squares(_res_fn, x0, loss="cauchy", f_scale=2.0, max_nfev=50)
            if sol.success and np.all(np.isfinite(sol.x[:3])):
                c_ref = sol.x[:3]
                depths_ref = ((pts_world - c_ref[None, :]) @ R_cw.T)[:, 2]
                if float(np.median(depths_ref)) > 0.05:
                    center = c_ref
        except Exception:
            pass
        return center

    def _estimate_scale_translation(
        self, P_gp1: np.ndarray, P_prior: np.ndarray, W: np.ndarray
    ) -> tuple[float, np.ndarray, float]:
        """Robustly estimate (scale, translation) with P_prior ~ scale * P_gp1 + translation.

        Uses 2-point RANSAC and a weighted least-squares refit. Returns the scale, translation, and inlier threshold.
        """
        num_anchors = len(P_gp1)
        prior_extent = float(np.linalg.norm(P_prior.max(axis=0) - P_prior.min(axis=0))) if num_anchors >= 2 else 0.0
        threshold = max(float(self.options.gp_ransac_inlier_threshold_m), 0.05 * prior_extent)

        if num_anchors == 1:
            scale = 1.0
            translation = P_prior[0] - P_gp1[0]
            inlier_mask = np.ones(1, dtype=bool)
        elif prior_extent < float(self.options.gp_alignment_min_extent_for_scale_m):
            # The anchors are too close together (e.g. a stationary camera) to observe the scale: keep the GP1
            # (metric depth) scale and estimate the translation only, by truncated voting and an inlier mean.
            scale = 1.0
            offsets = P_prior - P_gp1
            candidates = [offsets[i] for i in range(num_anchors)]
            if self.options.use_in_gp1:
                candidates.append(np.zeros(3, dtype=np.float64))
            scores = [
                np.sum(W * np.minimum(np.linalg.norm(offsets - c[None, :], axis=1), threshold)) for c in candidates
            ]
            translation = candidates[int(np.argmin(scores))]
            inliers = np.linalg.norm(offsets - translation[None, :], axis=1) < threshold
            if np.any(inliers):
                translation = np.sum(W[inliers, None] * offsets[inliers], axis=0) / np.sum(W[inliers])
        else:
            # Downweight spatially clustered stationary anchors so a stop at a traffic light
            # does not dominate RANSAC over moving segments of the trajectory.
            r_cell = max(1.0, 0.05 * prior_extent)
            dist_sq_mat = np.sum((P_prior[:, None, :] - P_prior[None, :, :]) ** 2, axis=2)
            density = np.sum(np.exp(-0.5 * dist_sq_mat / (r_cell**2)), axis=1)
            W_spatial = W / np.maximum(density, 1.0)

            min_baseline_sq = max(0.2, 0.15 * prior_extent) ** 2
            best_score = float("inf")
            best_scale = 1.0
            best_translation = np.mean(P_prior - P_gp1, axis=0)
            if self.options.use_in_gp1:
                id_errs = np.linalg.norm(P_gp1 - P_prior, axis=1)
                best_score = float(np.sum(W_spatial * np.minimum(id_errs, threshold)))
                best_scale = 1.0
                best_translation = np.zeros(3, dtype=np.float64)

            for min_b_sq in (min_baseline_sq, 0.04):
                found_pair = False
                for i in range(num_anchors):
                    for j in range(i + 1, num_anchors):
                        d_gp1 = P_gp1[j] - P_gp1[i]
                        d_prior = P_prior[j] - P_prior[i]
                        norm_gp1_sq = float(np.dot(d_gp1, d_gp1))
                        norm_prior_sq = float(np.dot(d_prior, d_prior))
                        if norm_gp1_sq < 0.01 or norm_prior_sq < min_b_sq:
                            continue
                        cand_scale = float(np.dot(d_prior, d_gp1) / norm_gp1_sq)
                        if cand_scale <= 0.05 or cand_scale >= 20.0:
                            continue
                        found_pair = True
                        wi, wj = W[i], W[j]
                        cand_trans = (
                            wi * (P_prior[i] - cand_scale * P_gp1[i]) + wj * (P_prior[j] - cand_scale * P_gp1[j])
                        ) / (wi + wj)
                        errs = np.linalg.norm(cand_scale * P_gp1 + cand_trans[None, :] - P_prior, axis=1)
                        score = float(np.sum(W_spatial * np.minimum(errs, threshold)))
                        if score < best_score:
                            best_score = score
                            best_scale = cand_scale
                            best_translation = cand_trans
                if found_pair:
                    break

            errs = np.linalg.norm(best_scale * P_gp1 + best_translation[None, :] - P_prior, axis=1)
            inlier_mask = errs < threshold
            if int(inlier_mask.sum()) >= 2:
                w_inl = W_spatial[inlier_mask]
                w_norm = (w_inl / np.sum(w_inl))[:, None]
                mu_gp1 = np.sum(w_norm * P_gp1[inlier_mask], axis=0)
                mu_prior = np.sum(w_norm * P_prior[inlier_mask], axis=0)
                gp1_zero = P_gp1[inlier_mask] - mu_gp1[None, :]
                prior_zero = P_prior[inlier_mask] - mu_prior[None, :]
                denom = float(np.sum(w_norm * (gp1_zero**2)))
                numer = float(np.sum(w_norm * (gp1_zero * prior_zero)))
                if denom > 1e-6 and 0.05 < (numer / denom) < 20.0:
                    refit_scale = numer / denom
                    refit_trans = mu_prior - refit_scale * mu_gp1
                    refit_errs = np.linalg.norm(refit_scale * P_gp1 + refit_trans[None, :] - P_prior, axis=1)
                    refit_score = float(np.sum(W_spatial * np.minimum(refit_errs, threshold)))
                    if refit_score <= best_score:
                        best_scale = refit_scale
                        best_translation = refit_trans
            scale = best_scale
            translation = best_translation

        return float(scale), np.asarray(translation, dtype=np.float64), threshold

    def align_gp1_to_location_priors_4dof(
        self,
        reconstruction: pycolmap.Reconstruction,
        dmap_scales: dict[int, float],
    ) -> dict[int, float]:
        """Align the GP1 reconstruction (centers, tracks, depth scales) to the prior frame via 4-DoF (scale, t).

        Since rotations are already in the prior orientation frame from Rotation Averaging,
        this estimates only a scale s > 0 (if >= 2 anchors) and a translation t in R^3
        from the resected 2D-3D ray centers using 2-point RANSAC + weighted least-squares.
        Parts of the reconstruction that are only weakly connected (e.g. across video cuts) are not constrained
        relative to each other in GP1, so each covisibility component is aligned to its own anchors.
        """
        if not self.options.enabled or not self.options.use_in_global_positioning:
            return dict(dmap_scales)

        # Covisibility components of the GP1 tracks, ignoring observations that GP1 could not explain.
        filtered = pycolmap.Reconstruction(reconstruction)
        pycolmap.ObservationManager(filtered).filter_points3D_with_large_reprojection_error(
            float(self.options.gp_alignment_max_angle_error_deg),
            filtered.point3D_ids(),
            pycolmap.ReprojectionErrorType.ANGULAR,
        )
        components = detect_covisibility_components(
            filtered,
            min_shared_points=int(self.options.gp_alignment_min_shared_points),
            min_component_size=1,
        )
        del filtered
        component_of = {image_id: index for index, component in enumerate(components) for image_id in component}

        anchors_by_component: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = {}
        for image_id in self.active_image_ids(reconstruction):
            anchor = self.anchors[image_id]
            c_prior = self._resect_camera_center_from_rays(reconstruction, anchor)
            if c_prior is None or image_id not in component_of:
                continue
            c_gp1 = np.asarray(reconstruction.image(image_id).projection_center(), dtype=np.float64)
            if not np.all(np.isfinite(c_gp1)):
                continue
            anchors_by_component.setdefault(component_of[image_id], []).append(
                (c_gp1, c_prior, max(float(anchor.confidence), 1e-3))
            )

        if not anchors_by_component:
            logger.warning("Location-prior 4-DoF GP1 alignment: no valid resected anchors; skipping alignment")
            return dict(dmap_scales)

        transforms: dict[int, tuple[float, np.ndarray]] = {}
        for index in sorted(anchors_by_component):
            P_gp1, P_prior, W = (np.asarray(values, dtype=np.float64) for values in zip(*anchors_by_component[index]))
            scale, translation, threshold = self._estimate_scale_translation(P_gp1, P_prior, W)
            final_errs = np.linalg.norm(scale * P_gp1 + translation[None, :] - P_prior, axis=1)
            logger.info(
                "Location-prior 4-DoF GP1 alignment%s: scale=%.4f, inliers=%d/%d (<%.1fm), median_err=%.3fm",
                (
                    f" of component {index + 1}/{len(components)} ({len(components[index])} images)"
                    if len(anchors_by_component) > 1
                    else ""
                ),
                scale,
                int((final_errs < threshold).sum()),
                len(P_gp1),
                threshold,
                float(np.median(final_errs)),
            )
            transforms[index] = (scale, translation)

        if len(transforms) == 1:
            # A single anchored part: transform all posed images and 3D points, c_new = scale * c_old + translation.
            ((scale, translation),) = transforms.values()
            reconstruction.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(), translation))
            return {int(img_id): float(val) * scale for img_id, val in dmap_scales.items()}

        # Transform each anchored component separately; components without anchors are left unchanged.
        image_scales = {}
        for index, (scale, translation) in transforms.items():
            for image_id in components[index]:
                image = reconstruction.image(image_id)
                pose = image.frame.rig_from_world
                center = scale * np.asarray(image.projection_center(), dtype=np.float64) + translation
                image.frame.rig_from_world = pycolmap.Rigid3d(pose.rotation, -(pose.rotation * center))
                image_scales[image_id] = scale
        for point3D_id in reconstruction.point3D_ids():
            point = reconstruction.point3D(point3D_id)
            votes = [component_of.get(element.image_id) for element in point.track.elements]
            votes = [index for index in votes if index is not None]
            if not votes:
                continue
            index = max(set(votes), key=votes.count)
            if index in transforms:
                scale, translation = transforms[index]
                point.xyz = scale * point.xyz + translation
        return {int(img_id): float(val) * image_scales.get(int(img_id), 1.0) for img_id, val in dmap_scales.items()}

    def append_gp_observations(
        self,
        problem,
        reconstruction: pycolmap.Reconstruction,
        centers: dict[int, np.ndarray],
        *,
        stage: str = "gp2",
    ):
        """Add bearing constraints from the anchor centers to the prior 3D points in Global Positioning.

        Returns the storage to keep alive while using the problem and the number of residuals.
        """
        if not self.options.enabled or not self.options.use_in_global_positioning:
            return [], 0
        if stage == "gp1" and not self.options.use_in_gp1:
            return [], 0

        storage = []
        num_residuals = 0
        kp_stddev_px = float(self.options.gp1_kp_stddev_px if stage == "gp1" else self.options.gp_kp_stddev_px)
        cauchy_scale = float(self.options.gp1_cauchy_scale) if stage == "gp1" else float(self.options.gp_cauchy_scale)
        base_weight = float(self.options.gp_weight)

        for image_id in self.active_image_ids(reconstruction):
            if image_id not in centers:
                continue
            anchor = self.anchors[image_id]
            image = reconstruction.image(image_id)
            camera = reconstruction.camera(image.camera_id)
            fx, fy = float(camera.focal_length_x), float(camera.focal_length_y)
            xy_px = anchor.uv_norm * np.array([float(camera.width), float(camera.height)], dtype=np.float64)
            cam_pts = np.asarray(camera.cam_from_img(xy_px), dtype=np.float64)
            if cam_pts.ndim != 2 or len(cam_pts) == 0:
                continue
            bearings = np.column_stack([cam_pts, np.ones(len(cam_pts), dtype=np.float64)])
            norms = np.linalg.norm(bearings, axis=1, keepdims=True)
            valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-9)
            if not np.any(valid):
                continue
            bearings = bearings[valid] / norms[valid]
            angular_stddevs_2d = kp_stddev_px * np.ones((len(bearings), 1), dtype=np.float64) / np.array([fx, fy])
            bearing_covars = _angular_stds_to_xyz_covar(bearings, angular_stddevs_2d)
            angular_stddevs = np.sqrt(np.clip(np.diagonal(bearing_covars, axis1=1, axis2=2)[:, :2], 1e-18, None))
            angular_stddevs = np.maximum(angular_stddevs, 1e-9)
            stddevs = np.column_stack([angular_stddevs, angular_stddevs.mean(axis=1)])

            loss = pyceres.LossFunction(
                dict(name="cauchy", params=[cauchy_scale], magnitude=base_weight * float(anchor.confidence))
            )
            observations, count = gp_costs.append_bearing_observations(
                problem,
                centers[image_id],
                np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float64),
                np.ascontiguousarray(anchor.points3D_xyz[valid], dtype=np.float64),
                np.ascontiguousarray(bearings, dtype=np.float64),
                np.ascontiguousarray(stddevs, dtype=np.float64),
                loss,
            )
            storage.append(observations)
            num_residuals += count

        logger.info(
            "Added %d location-prior 2D-3D positioning observations across %d active anchors",
            num_residuals,
            len(self.active_image_ids(reconstruction)),
        )
        return storage, num_residuals

    def append_ba_constraints(self, problem, reconstruction: pycolmap.Reconstruction, image_ids):
        """Add reprojection residuals of the prior 3D points into the anchors in Bundle Adjustment.

        Returns the storage to keep alive while using the problem and the number of residuals.
        """
        if not self.options.enabled or not self.options.use_in_bundle_adjustment:
            return [], 0

        storage = []
        num_residuals = 0
        kp_stddev_px = float(self.options.ba_kp_stddev_px)
        loss_type = getattr(pycolmap.LossFunctionType, self.options.ba_loss_name.upper())
        loss_scale_px = float(self.options.ba_loss_scale_px)
        base_weight = float(self.options.ba_weight)

        image_ids = set(image_ids)
        for image_id in self.active_image_ids(reconstruction):
            if image_id not in image_ids:
                continue
            image = reconstruction.image(image_id)
            anchor = self.anchors[image_id]
            camera = reconstruction.camera(image.camera_id)
            xy_px = anchor.uv_norm * np.array([float(camera.width), float(camera.height)], dtype=np.float64)

            # Filter out points currently behind the camera to avoid degenerate projection Jacobians
            pts_cam = image.cam_from_world() * anchor.points3D_xyz
            in_front = pts_cam[:, 2] > 0.05
            if not np.any(in_front):
                continue

            loss, count = ba_costs.append_constant_point_reprojections(
                problem,
                reconstruction,
                image_id,
                np.ascontiguousarray(xy_px[in_front], dtype=np.float64),
                np.ascontiguousarray(anchor.points3D_xyz[in_front], dtype=np.float64),
                loss_type,
                loss_scale_px,
                base_weight * float(anchor.confidence) / (kp_stddev_px**2),
            )
            storage.append(loss)
            num_residuals += count

        return storage, num_residuals


def load_location_priors(
    options: LocationPriorOptions,
    solve_state: SolveState,
) -> LocationPriorSet | None:
    """Load location priors from an .npz file and match them to images in solve_state.

    The priors come from any absolute localization of some frames against a georeferenced or otherwise fixed world
    frame (the reconstruction is then expressed in that frame). Fields, for M prior frames, P world points and K
    2D-3D matches:
        image_names          (M,) str     image file names (matched by basename, else by timestamp-like stem)
        confidence           (M,) float   in [0, 1]; scales the prior weights, frames below min_confidence are skipped
        R_cam_from_world     (M, 3, 3)    absolute camera rotations
        cov_cam_from_world   (M, 6, 6)    pose covariance [rotation (rad), translation] in the COLMAP cam_from_world
                                          tangent space; only the rotation block is used
        points3D             (P, 3)       world points observed by the prior frames
        match_indptr         (M + 1,)     CSR offsets of each frame's matches
        match_point_indices  (K,)         index into points3D of each match
        match_uv_norm        (K, 2)       image coordinates normalized by the image size, (x / width, y / height)
    """
    if not options.enabled or not options.path:
        return None

    npz_path = Path(options.path).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Location priors file does not exist: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    image_names = [str(name) for name in data["image_names"]]
    confidences = np.asarray(data["confidence"], dtype=np.float64)
    R_cam_from_world = np.asarray(data["R_cam_from_world"], dtype=np.float64)
    cov_cam_from_world = np.asarray(data["cov_cam_from_world"], dtype=np.float64)
    points3D = np.asarray(data["points3D"], dtype=np.float64)
    match_indptr = np.asarray(data["match_indptr"], dtype=np.int64)
    match_point_indices = np.asarray(data["match_point_indices"], dtype=np.uint32)
    match_uv_norm = np.asarray(data["match_uv_norm"], dtype=np.float64)

    # Map solve_state images by exact basename and by microsecond timestamp
    id_by_basename: dict[str, int] = {}
    id_by_ts_us: dict[int, int] = {}
    for image_id, image in solve_state.reconstruction.images.items():
        basename = Path(image.name).name
        id_by_basename[basename] = int(image_id)
        ts_us = _parse_timestamp_us(basename)
        if ts_us is not None:
            id_by_ts_us[ts_us] = int(image_id)

    anchors: dict[int, LocationAnchorPrior] = {}
    skipped_low_conf = 0
    skipped_unmatched = 0
    for idx, prior_name in enumerate(image_names):
        conf = float(confidences[idx])
        start, end = int(match_indptr[idx]), int(match_indptr[idx + 1])
        num_matches = end - start
        if conf < options.min_confidence or num_matches < options.min_inliers:
            skipped_low_conf += 1
            continue

        basename = Path(prior_name).name
        matched_id = id_by_basename.get(basename)
        if matched_id is None:
            ts_us = _parse_timestamp_us(basename)
            if ts_us is not None and id_by_ts_us:
                closest_ts = min(id_by_ts_us.keys(), key=lambda k: abs(k - ts_us))
                if abs(closest_ts - ts_us) <= 1000:
                    matched_id = id_by_ts_us[closest_ts]
        if matched_id is None:
            skipped_unmatched += 1
            continue

        pt_ids = match_point_indices[start:end].copy()
        pts_xyz = points3D[pt_ids].copy()
        uv_norm = match_uv_norm[start:end].copy()

        anchors[matched_id] = LocationAnchorPrior(
            image_id=matched_id,
            image_name=basename,
            confidence=conf,
            R_cam_from_world=R_cam_from_world[idx].copy(),
            cov_rot_cam_from_world=cov_cam_from_world[idx, :3, :3].copy(),
            prior_point_ids=pt_ids,
            points3D_xyz=pts_xyz,
            uv_norm=uv_norm,
        )

    logger.info(
        "Loaded location priors from %s: %d matched keyframes (%d skipped low-conf/inliers, %d unmatched)",
        npz_path.name,
        len(anchors),
        skipped_low_conf,
        skipped_unmatched,
    )
    return LocationPriorSet(
        options=options,
        anchors=anchors,
        mapped_points_xyz=points3D,
    )

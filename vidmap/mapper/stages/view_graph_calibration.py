"""View-graph calibration and pair restoration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pycolmap

from vidmap.mapper.focal_prior import native_focal_priors
from vidmap.mapper.native.extension import native
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.view_graph import VGCCalibrationOptions

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ViewGraphCalibrator:
    """Calibrate the view graph."""

    solve_state: SolveState
    options: VGCCalibrationOptions
    enabled: bool
    consecutive_pair_ids: list[int]
    exclusion_ids: set[int]
    focal_prior: dict[int, tuple[tuple[float, float], ...]] | None = None

    def calibrate(self) -> None:
        rec = self.solve_state.reconstruction
        cameras = rec.cameras
        state = self.solve_state

        if not self.enabled:
            return

        logger.info("Running view graph calibration...")

        if self.options.unlock_focal:
            for _cid, _cam in cameras.items():
                _cam.has_prior_focal_length = False

        original_configs = {}
        try:
            if self.exclusion_ids:
                for pid in state.pair_order:
                    pair = state.pair_data(pid)
                    if pid in self.exclusion_ids:
                        original_configs[pid] = pair.geometry.config
                        geometry = pair.geometry
                        geometry.config = pycolmap.TwoViewGeometryConfiguration.PLANAR
                        pair.geometry = geometry
                logger.info(
                    "Temporarily marked %d pairs as PLANAR for two-stage VGC",
                    len(original_configs),
                )

            consec_validity_before_vgc = {pid: state.pose_graph.is_valid(pid) for pid in self.consecutive_pair_ids}

            vgc_options = pycolmap.ViewGraphCalibrationOptions()
            focal_priors = []
            num_inputs = 0
            valid_configurations = {
                pycolmap.TwoViewGeometryConfiguration.CALIBRATED,
                pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED,
            }
            for pid in state.pair_order:
                pair = state.pair_data(pid)
                geometry = pair.geometry
                if geometry.config not in valid_configurations:
                    continue
                if not state.pose_graph.is_valid(pid):
                    continue
                if geometry.F is None:
                    raise RuntimeError(
                        f"Valid VGC pair {pid} ({pycolmap.pair_id_to_image_pair(pid)}) has no fundamental matrix"
                    )
                num_inputs += 1

            if self.focal_prior is not None:
                prior = self.focal_prior
                vgc_options.min_focal_length_ratio = np.finfo(float).tiny
                vgc_options.max_focal_length_ratio = np.finfo(float).max
                weight = self.options.focal_prior_weight
                if self.options.normalize_weight_by_pair_count and num_inputs and prior:
                    weight *= num_inputs / sum(len(rows) for rows in prior.values())
                focal_priors = native_focal_priors(
                    prior,
                    camera_ids=prior,
                    loss="cauchy",
                    weight=weight,
                )
            invalid_count = native.calibrate_focal_lengths(
                vgc_options,
                state.reconstruction,
                state.pose_graph,
                state.sidecars,
                focal_priors,
            )
            logger.info(
                "VGC: invalidated %d / %d pairs (residual^2 > %.4f)",
                invalid_count,
                num_inputs,
                vgc_options.max_calibration_error**2,
            )
        finally:
            if original_configs:
                logger.info(
                    "Restoring %d pair configurations after two-stage VGC",
                    len(original_configs),
                )
                for pid in state.pair_order:
                    pair = state.pair_data(pid)
                    if pid in original_configs:
                        geometry = pair.geometry
                        geometry.config = original_configs[pid]
                        pair.geometry = geometry
                logger.info(
                    "Restored %d pair configurations for rotation averaging",
                    len(original_configs),
                )

        restored_consec_count = 0
        for pid in self.consecutive_pair_ids:
            if consec_validity_before_vgc[pid] and not state.pose_graph.is_valid(pid):
                state.pose_graph.set_valid_edge(pid)
                restored_consec_count += 1
        if restored_consec_count > 0:
            logger.info(
                "Restored %d consecutive pairs invalidated by VGC",
                restored_consec_count,
            )

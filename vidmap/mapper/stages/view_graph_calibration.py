"""View-graph calibration and pair restoration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pycolmap

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
        state.import_cameras()

        original_configs = {}
        try:
            if self.exclusion_ids:
                for pid, pair in state.pair_records().items():
                    if pid in self.exclusion_ids:
                        original_configs[pid] = pair.geometry.configuration
                        geometry = pair.geometry
                        geometry.configuration = pycolmap.TwoViewGeometryConfiguration.PLANAR
                        pair.geometry = geometry
                        state.update_pair(pair)
                logger.info(
                    "Temporarily marked %d pairs as PLANAR for two-stage VGC",
                    len(original_configs),
                )

            consec_validity_before_vgc = {pid: state.pair(pid).is_valid for pid in self.consecutive_pair_ids}

            vgc_options = native.FocalCalibrationOptions()
            num_inputs = 0
            valid_configurations = {
                pycolmap.TwoViewGeometryConfiguration.CALIBRATED,
                pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED,
            }
            for pid, pair in state.pair_records().items():
                geometry = pair.geometry
                if geometry.configuration not in valid_configurations:
                    continue
                if not pair.is_valid:
                    continue
                if not geometry.has_fundamental:
                    raise RuntimeError(
                        f"Valid VGC pair {pid} ({pair.image_id1}->{pair.image_id2}) has no fundamental matrix"
                    )
                num_inputs += 1

            result = native.calibrate_focal_lengths(vgc_options, state.native_problem)
            if not result.success:
                raise RuntimeError("View graph calibration failed")

            invalid_count = native.apply_focal_calibration(
                vgc_options,
                result,
                state.native_problem,
            )
            state.export_cameras()
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
                for pid, pair in state.pair_records().items():
                    if pid in original_configs:
                        geometry = pair.geometry
                        geometry.configuration = original_configs[pid]
                        pair.geometry = geometry
                        state.update_pair(pair)
                logger.info(
                    "Restored %d pair configurations for rotation averaging",
                    len(original_configs),
                )

        restored_consec_count = 0
        for pid in self.consecutive_pair_ids:
            pair = state.pair(pid)
            if consec_validity_before_vgc[pid] and not pair.is_valid:
                pair.is_valid = True
                state.update_pair(pair)
                restored_consec_count += 1
        if restored_consec_count > 0:
            logger.info(
                "Restored %d consecutive pairs invalidated by VGC",
                restored_consec_count,
            )

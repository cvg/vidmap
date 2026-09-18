"""Build per-image focal priors from original calibration predictions."""

import h5py
import numpy as np


def native_focal_priors(prior, *, camera_ids, loss, scale=1.0, weight=1.0):
    from vidmap.mapper.native.extension import native
    from vidmap.mapper.native.losses import build_named_loss_config

    records = []
    for camera_id in sorted(set(camera_ids)):
        record = native.LogFocalPriorRecord()
        record.camera_id = camera_id
        record.observations = np.asarray(prior[camera_id], dtype=np.float64)
        record.loss = build_named_loss_config(loss, scale=scale, weight=weight)
        records.append(record)
    return records


def load_focal_prior(path, state, *, log_focal_stddev=None, shared=False):
    images = state.reconstruction.images
    cameras = state.reconstruction.cameras
    if shared and len(cameras) != 1:
        raise ValueError("Selected shared calibration requires one camera")
    associations = (
        [("batch_calibration", next(iter(cameras)))]
        if shared
        else [(images[i].name, images[i].camera_id) for i in state.image_order]
    )
    if not associations:
        raise ValueError("Calibration results require camera associations")
    observations = {}
    with h5py.File(path, "r") as hfile:
        for name, camera_id in associations:
            result = hfile[name]
            camera = cameras[camera_id]
            if tuple(result["image_size"][:]) != (camera.width, camera.height):
                raise ValueError("Calibration dimensions disagree with database")
            K = np.asarray(result["K"], dtype=np.float32)
            focal = float((K[0, 0] + K[1, 1]) / 2.0)
            sigma_log_focal = log_focal_stddev
            if sigma_log_focal is None:
                std = result["focal_std_px"][:]
                if not shared and len(std) != 1:
                    raise ValueError("Per-view uncertainty requires one source-pixel standard deviation")
                sigma_log_focal = float(min(std)) / focal
            observations.setdefault(camera_id, []).append((focal, sigma_log_focal))
    return {cid: tuple(rows) for cid, rows in observations.items()}

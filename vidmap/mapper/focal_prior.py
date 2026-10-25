"""Build per-image focal priors from original calibration predictions."""

import h5py
import numpy as np


def native_focal_priors(prior, *, camera_ids, loss, scale=1.0, weight=1.0):
    import pycolmap

    from vidmap.mapper.native.extension import native

    records = []
    for camera_id in sorted(set(camera_ids)):
        record = native.LogFocalPriorRecord()
        record.camera_id = camera_id
        record.observations = np.asarray(prior[camera_id], dtype=np.float64)
        record.loss = pycolmap.create_ceres_loss_function(pycolmap.LossFunctionType(loss.upper()), scale, weight)
        records.append(record)
    return records


def load_focal_prior(path, state, evidence, *, log_focal_stddev=None):
    cameras = state.reconstruction.cameras
    images = {image.name: image for image in state.reconstruction.images.values()}
    shared = evidence.inference == "selected_batch"
    ordered = evidence.observations
    if not shared:
        by_name = {observation.image_name: observation for observation in ordered}
        ordered = tuple(by_name.pop(state.reconstruction.images[i].name) for i in state.image_order)
        if by_name or len(ordered) != len(evidence.observations):
            raise ValueError("Per-view calibration observations disagree with database images")
    observations = {}
    with h5py.File(path, "r") as hfile:
        for observation in ordered:
            camera_id = observation.camera_id
            if observation.image_name is not None and images[observation.image_name].camera_id != camera_id:
                raise ValueError("Calibration observation disagrees with database camera membership")
            result = hfile[observation.key]
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

"""Structural fingerprints for native mapper records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pycolmap


def _hash_array(value: Any, *, length: int = 12) -> str:
    if value is None:
        return ""
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha1(array.tobytes()).hexdigest()[:length]


def _pycolmap_pose(record: Any) -> pycolmap.Rigid3d:
    return pycolmap.Rigid3d(
        rotation=np.asarray(record.rotation_xyzw, dtype=np.float64),
        translation=np.asarray(record.translation, dtype=np.float64),
    )


def fingerprint_cameras(cameras: Mapping[int, Any] | None) -> dict[str, Any]:
    items = sorted((cameras or {}).items(), key=lambda item: int(item[0]))
    total = hashlib.sha1()
    identifiers = hashlib.sha1()
    models = hashlib.sha1()
    parameters = hashlib.sha1()
    for camera_id, camera in items:
        identifier = int(camera_id).to_bytes(8, "little", signed=False)
        model = str(camera.model).encode()
        params = _hash_array(camera.params, length=16).encode()
        total.update(identifier)
        total.update(model)
        total.update(params)
        identifiers.update(identifier)
        models.update(model)
        parameters.update(params)
    return {
        "num_cameras": len(items),
        "content_hash": total.hexdigest()[:16],
        "per_field_hash": {
            "camera_id": identifiers.hexdigest()[:12],
            "model": models.hexdigest()[:12],
            "params": parameters.hexdigest()[:12],
        },
    }


def fingerprint_tracks(tracks: Mapping[int, Any]) -> dict[str, Any]:
    items = sorted(tracks.items(), key=lambda item: int(item[0]))
    total = hashlib.sha1()
    field_hashes = {name: hashlib.sha1() for name in ("track_id", "xyz", "color", "error", "obs", "lc")}
    identities = []
    lc_identities = []
    per_image_observations: dict[int, int] = {}
    observed_image_ids: set[int] = set()
    total_observations = 0
    total_lc_observations = 0
    sample = []

    for track_id, track in items:
        observations = [(int(image_id), int(point2D_idx)) for image_id, point2D_idx in track.observations]
        loop_closures = [
            (int(image_id), int(point2D_idx)) for image_id, point2D_idx in track.loop_closure_observations
        ]
        total_observations += len(observations)
        total_lc_observations += len(loop_closures)
        for image_id, _ in observations:
            observed_image_ids.add(image_id)
            per_image_observations[image_id] = per_image_observations.get(image_id, 0) + 1
        observed_image_ids.update(image_id for image_id, _ in loop_closures)

        identifier = int(track_id).to_bytes(8, "little", signed=False)
        sorted_observations = sorted(observations)
        sorted_loop_closures = sorted(loop_closures)
        field_hashes["track_id"].update(identifier)
        field_hashes["xyz"].update(_hash_array(track.xyz).encode())
        field_hashes["color"].update(_hash_array(track.color).encode())
        field_hashes["error"].update(repr(float(track.error)).encode())
        field_hashes["obs"].update(_hash_array(sorted_observations).encode())
        field_hashes["lc"].update(_hash_array(sorted_loop_closures).encode())
        identities.append(json.dumps(sorted_observations, separators=(",", ":")))
        lc_identities.append(json.dumps(sorted_loop_closures, separators=(",", ":")))

        content = b"".join(
            (
                identifier,
                identifier,
                _hash_array(track.xyz).encode(),
                _hash_array(sorted_observations).encode(),
                _hash_array(sorted_loop_closures).encode(),
            )
        )
        # The replay fingerprint schema includes traversal and content identities.
        total.update(content)
        if len(sample) < 3:
            sample.append({"track_id": int(track_id)})

    return {
        "n_tracks": len(items),
        "total_observations": total_observations,
        "total_lc_observations": total_lc_observations,
        "observed_image_count": len(observed_image_ids),
        "content_hash": total.hexdigest()[:16],
        "per_field_hash": {
            "track_id": field_hashes["track_id"].hexdigest()[:12],
            "xyz": field_hashes["xyz"].hexdigest()[:12],
            "color": field_hashes["color"].hexdigest()[:12],
            "error": field_hashes["error"].hexdigest()[:12],
            "observations": field_hashes["obs"].hexdigest()[:12],
            "lc_observations": field_hashes["lc"].hexdigest()[:12],
            "track_identity": hashlib.sha1("\n".join(sorted(identities)).encode()).hexdigest()[:12],
            "lc_track_identity": hashlib.sha1("\n".join(sorted(lc_identities)).encode()).hexdigest()[:12],
            "per_image_observation_counts": hashlib.sha1(
                json.dumps(sorted(per_image_observations.items()), separators=(",", ":")).encode()
            ).hexdigest()[:12],
        },
        "sample": sample,
    }


_IMAGE_ARRAYS = {
    "features": "keypoints",
    "features_undist": "bearings",
    "angular_stddevs": "angular_stddevs",
    "depth_priors": "depth_values",
    "depth_prior_stddevs": "depth_stddevs",
    "depth_prior_validity": "depth_validity",
    "is_inlier": "is_inlier",
    "is_depth_outlier": "is_depth_outlier",
    "is_track_anchor": "is_track_anchor",
}


def fingerprint_images(images: Mapping[int, Any]) -> dict[str, Any]:
    items = sorted(images.items(), key=lambda item: int(item[0]))
    total = hashlib.sha1()
    fields = {
        name: hashlib.sha1()
        for name in (
            "image_id",
            "name",
            "has_pose",
            "translation",
            "quaternion",
            "matrix",
            "center",
            *_IMAGE_ARRAYS,
        )
    }
    total_features = 0
    total_depth_priors = 0
    registered = 0
    sample = []

    def update(name: str, value: bytes) -> None:
        fields[name].update(value)
        total.update(value)

    for image_id, image in items:
        pose = image.pose if image.pose.has_pose else None
        registered += int(pose is not None)
        total_features += int(np.asarray(image.keypoints).shape[0])
        total_depth_priors += int(np.asarray(image.depth_values).shape[0])
        update("image_id", int(image_id).to_bytes(8, "little", signed=False))
        update("has_pose", b"\x01" if pose is not None else b"\x00")
        update("name", image.name.encode())
        for output_name, attribute in _IMAGE_ARRAYS.items():
            update(output_name, _hash_array(getattr(image, attribute)).encode())

        if pose is None:
            for name in ("translation", "quaternion", "matrix", "center"):
                update(name, b"")
        else:
            pycolmap_pose = _pycolmap_pose(pose)
            matrix = np.asarray(pycolmap_pose.rotation.matrix(), dtype=np.float64)
            update("translation", _hash_array(pose.translation).encode())
            update("quaternion", _hash_array(pose.rotation_xyzw).encode())
            update("matrix", _hash_array(matrix).encode())
            update("center", _hash_array(pycolmap_pose.inverse().translation).encode())
        if len(sample) < 3:
            sample.append({"image_id": int(image_id)})

    per_field_hash = {
        "image_id": fields["image_id"].hexdigest()[:12],
        "name": fields["name"].hexdigest()[:12],
        "has_pose": fields["has_pose"].hexdigest()[:12],
        "cam_from_world.translation": fields["translation"].hexdigest()[:12],
        "cam_from_world.rotation.quat": fields["quaternion"].hexdigest()[:12],
        "cam_from_world.rotation.matrix": fields["matrix"].hexdigest()[:12],
        "solver_internal.center": fields["center"].hexdigest()[:12],
    }
    per_field_hash.update({name: fields[name].hexdigest()[:12] for name in _IMAGE_ARRAYS})
    return {
        "n_images": len(items),
        "n_registered": registered,
        "total_features": total_features,
        "total_depth_priors": total_depth_priors,
        "content_hash": total.hexdigest()[:16],
        "per_field_hash": per_field_hash,
        "sample": sample,
    }


def fingerprint_scene(
    images: Mapping[int, Any], tracks: Mapping[int, Any], cameras: Mapping[int, Any]
) -> dict[str, Any]:
    image_fingerprint = fingerprint_images(images)
    track_fingerprint = fingerprint_tracks(tracks)
    total = hashlib.sha1()
    total.update(image_fingerprint["content_hash"].encode())
    total.update(track_fingerprint["content_hash"].encode())
    return {
        "n_images": image_fingerprint["n_images"],
        "n_registered": image_fingerprint["n_registered"],
        "n_tracks": track_fingerprint["n_tracks"],
        "total_observations": track_fingerprint["total_observations"],
        "total_lc_observations": track_fingerprint["total_lc_observations"],
        "content_hash": total.hexdigest()[:16],
        "cameras": fingerprint_cameras(cameras),
        "images_per_field_hash": image_fingerprint["per_field_hash"],
        "tracks_per_field_hash": track_fingerprint["per_field_hash"],
    }

"""Canonical scene summaries for mapper byte evidence."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np
import pycolmap

from .canonical import (
    count_content_hash,
    hash_array_like,
    hash_pose_translation,
    image_has_pose,
    image_pose,
    image_value,
    inlier_array,
    int_sort_key,
    pair_cam2_from_cam1,
    pose_rotation,
    rotation_matrix,
    rotation_quaternion,
    stable_tuple_json_hash,
    update_intish_hash,
)


def cameras_summary(cameras: dict | None) -> dict:
    h = hashlib.sha1()
    items = () if cameras is None else cameras.items()
    for camera_id, camera in sorted(items, key=lambda item: int(item[0])):
        h.update(int(camera_id).to_bytes(8, "little", signed=False))
        h.update(_camera_model_id(camera).to_bytes(4, "little", signed=True))
        h.update(int(camera.width).to_bytes(8, "little", signed=False))
        h.update(int(camera.height).to_bytes(8, "little", signed=False))
        h.update(_camera_params_hash(camera.params).encode())
    return {
        "num_cameras": 0 if cameras is None else len(cameras),
        "content_hash": h.hexdigest()[:16],
    }


def _camera_params_hash(params: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(params, dtype=np.float64))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def _camera_model_id(camera: Any) -> int:
    if isinstance(camera, pycolmap.Camera):
        return int(camera.model)
    return int(camera.model_id)


def native_problem_summary(state: Any) -> dict:
    items = sorted(state.pair_records().items(), key=lambda item: int(item[0]))
    n_pairs = 0
    n_valid = 0
    total_matches = 0
    total_inliers = 0
    total_lc = 0
    sample = []

    for pid, pair in items:
        n_pairs += 1
        valid = bool(pair.is_valid)
        n_valid += int(valid)
        matches = np.asarray(pair.all_matches)
        inliers = inlier_array(pair)
        are_lc = np.asarray(pair.are_loop_closure)
        total_matches += int(matches.shape[0])
        total_inliers += int(len(inliers))
        total_lc += int(sum(1 for x in are_lc if x))
        if len(sample) < 3:
            sample.append({"pair_id": int(pid), "is_valid": valid})
    return {
        "n_pairs": n_pairs,
        "n_valid": n_valid,
        "total_matches": total_matches,
        "total_inliers": total_inliers,
        "total_are_lc_true": total_lc,
        "content_hash": count_content_hash(n_pairs, n_valid, total_matches, total_inliers, total_lc),
        "sample": sample,
    }


_IMAGE_ARRAY_FIELDS = (
    "features",
    "features_undist",
    "angular_stddevs",
    "depth_priors",
    "depth_prior_stddevs",
    "depth_prior_validity",
    "is_inlier",
    "is_depth_outlier",
    "is_track_anchor",
)


def images_summary(images: dict) -> dict:
    items = sorted(images.items(), key=lambda item: int(item[0]))
    n_images = 0
    total_features = 0
    total_depth_priors = 0
    n_registered = 0
    sample = []
    h_total = hashlib.sha1()
    h_id = hashlib.sha1()
    h_name = hashlib.sha1()
    h_has_pose = hashlib.sha1()
    h_cfw_t = hashlib.sha1()
    h_cfw_q = hashlib.sha1()
    h_cfw_matrix = hashlib.sha1()
    h_solver_center = hashlib.sha1()
    h_arrays = {field: hashlib.sha1() for field in _IMAGE_ARRAY_FIELDS}

    def bump(hasher, blob):
        hasher.update(blob)
        h_total.update(blob)

    for iid, image in items:
        n_images += 1
        has_pose = image_has_pose(image)
        n_registered += int(has_pose)
        features = np.asarray(image_value(image, "features"))
        depth_priors = np.asarray(image_value(image, "depth_priors"))
        total_features += int(features.shape[0]) if features.ndim else 0
        total_depth_priors += int(depth_priors.shape[0]) if depth_priors.ndim else 0

        bump(h_id, int(iid).to_bytes(8, "little", signed=False))
        bump(h_has_pose, b"\x01" if has_pose else b"\x00")
        bump(h_name, str(image.name).encode())

        for field in _IMAGE_ARRAY_FIELDS:
            bump(
                h_arrays[field],
                hash_array_like(image_value(image, field)).encode(),
            )

        pose = image_pose(image)
        if pose is None:
            bump(h_cfw_t, hash_array_like(None).encode())
            bump(h_cfw_q, hash_array_like(None).encode())
            bump(h_cfw_matrix, hash_array_like(None).encode())
            bump(h_solver_center, hash_array_like(None).encode())
        else:
            rotation = pose_rotation(pose)
            translation = pose.translation
            bump(
                h_cfw_t,
                hash_pose_translation(translation).encode(),
            )
            bump(h_cfw_q, hash_array_like(rotation_quaternion(rotation)).encode())
            bump(
                h_cfw_matrix,
                hash_array_like(rotation_matrix(rotation)).encode(),
            )
            bump(
                h_solver_center,
                hash_pose_translation(translation).encode(),
            )

        if len(sample) < 3:
            sample.append({"image_id": int(iid)})

    per_field_hash = {
        "image_id": h_id.hexdigest()[:12],
        "name": h_name.hexdigest()[:12],
        "has_pose": h_has_pose.hexdigest()[:12],
        "cam_from_world.translation": h_cfw_t.hexdigest()[:12],
        "cam_from_world.rotation.quat": h_cfw_q.hexdigest()[:12],
        "cam_from_world.rotation.matrix": h_cfw_matrix.hexdigest()[:12],
        "solver_internal.center": h_solver_center.hexdigest()[:12],
    }
    per_field_hash.update({field: h_arrays[field].hexdigest()[:12] for field in _IMAGE_ARRAY_FIELDS})
    return {
        "n_images": n_images,
        "n_registered": n_registered,
        "total_features": total_features,
        "total_depth_priors": total_depth_priors,
        "content_hash": h_total.hexdigest()[:16],
        "per_field_hash": per_field_hash,
        "sample": sample,
    }


def track_observation_lists(track: Any) -> tuple[list, list]:
    return (
        list(ordered_observation_pairs(track.observations)),
        list(ordered_observation_pairs(track.loop_closure_observations)),
    )


def ordered_observation_pairs(observations: list) -> tuple[tuple[int, int], ...]:
    return tuple((int(image_id), int(point2D_idx)) for image_id, point2D_idx in observations)


def _canonical_observation_pairs(
    observations: list,
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(ordered_observation_pairs(observations)))


def track_identity_map(tracks: dict | None) -> dict:
    identity_by_point3D_id = {}
    for point3D_id, track in sorted((tracks or {}).items(), key=lambda item: int_sort_key(item[0])):
        obs_list, lc_list = track_observation_lists(track)
        identity_by_point3D_id[int(point3D_id)] = {
            "observations": _canonical_observation_pairs(obs_list),
            "lc_observations": _canonical_observation_pairs(lc_list),
        }
    return identity_by_point3D_id


def track_identity_summary(identity_by_point3D_id: dict | None) -> dict:
    h = hashlib.sha1()
    count = 0
    total_observations = 0
    total_lc_observations = 0
    sample = []
    for point3D_id, identity in sorted(
        (() if identity_by_point3D_id is None else identity_by_point3D_id.items()),
        key=lambda item: int_sort_key(item[0]),
    ):
        observations = tuple(identity["observations"])
        lc_observations = tuple(identity["lc_observations"])
        count += 1
        total_observations += len(observations)
        total_lc_observations += len(lc_observations)
        h.update(
            json.dumps(
                [int(point3D_id), observations, lc_observations],
                separators=(",", ":"),
            ).encode()
        )
        if len(sample) < 3:
            sample.append(
                {
                    "point3D_id": int(point3D_id),
                    "observations": len(observations),
                    "lc_observations": len(lc_observations),
                }
            )
    return {
        "count": count,
        "total_observations": total_observations,
        "total_lc_observations": total_lc_observations,
        "content_hash": h.hexdigest()[:16],
        "sample": sample,
    }


def track_records_summary(tracks: dict | None) -> dict:
    items = sorted((tracks or {}).items(), key=lambda item: int(item[0]))
    n_tracks = 0
    total_obs = 0
    total_lc_obs = 0
    observed_image_ids = set()
    per_image_observations: defaultdict[int, int] = defaultdict(int)
    sample = []
    h = hashlib.sha1()
    h_track_id = hashlib.sha1()
    h_track_id_order = hashlib.sha1()
    h_xyz = hashlib.sha1()
    h_color = hashlib.sha1()
    h_error = hashlib.sha1()
    h_obs = hashlib.sha1()
    h_obs_order = hashlib.sha1()
    h_lc_obs = hashlib.sha1()
    h_lc_obs_order = hashlib.sha1()

    for tid in sorted((tracks or {}).keys(), key=int):
        update_intish_hash(h_track_id_order, tid)

    for tid, track in items:
        n_tracks += 1
        obs_list, lc_list = track_observation_lists(track)
        total_obs += len(obs_list)
        total_lc_obs += len(lc_list)
        for image_id, _ in obs_list:
            observed_image_ids.add(int(image_id))
            per_image_observations[int(image_id)] += 1
        for image_id, _ in lc_list:
            observed_image_ids.add(int(image_id))
        sorted_obs = sorted(obs_list)
        sorted_lc = sorted(lc_list)
        ordered_obs = [(int(tid), int(i), int(iid), int(pid)) for i, (iid, pid) in enumerate(obs_list)]
        ordered_lc = [(int(tid), int(i), int(iid), int(pid)) for i, (iid, pid) in enumerate(lc_list)]

        update_intish_hash(h, tid)
        h.update(hash_array_like(track.xyz).encode())
        h.update(hash_array_like(sorted_obs).encode())
        h.update(hash_array_like(sorted_lc).encode())

        update_intish_hash(h_track_id, tid)
        h_xyz.update(hash_array_like(track.xyz).encode())
        h_color.update(b"")
        h_error.update(b"")
        h_obs.update(hash_array_like(sorted_obs).encode())
        h_lc_obs.update(hash_array_like(sorted_lc).encode())
        h_obs_order.update(stable_tuple_json_hash(ordered_obs).encode())
        h_lc_obs_order.update(stable_tuple_json_hash(ordered_lc).encode())

        update_intish_hash(h, tid)
        h.update(hash_array_like(track.xyz).encode())
        h.update(hash_array_like(sorted_obs).encode())
        h.update(hash_array_like(sorted_lc).encode())

        if len(sample) < 3:
            sample.append({"track_id": int(tid)})
    return {
        "n_tracks": n_tracks,
        "total_observations": total_obs,
        "total_lc_observations": total_lc_obs,
        "observed_image_count": len(observed_image_ids),
        "content_hash": h.hexdigest()[:16],
        "per_field_hash": {
            "track_id": h_track_id.hexdigest()[:12],
            "track_id_order": h_track_id_order.hexdigest()[:12],
            "xyz": h_xyz.hexdigest()[:12],
            "color": h_color.hexdigest()[:12],
            "error": h_error.hexdigest()[:12],
            "observations": h_obs.hexdigest()[:12],
            "observation_tuples_ordered": h_obs_order.hexdigest()[:12],
            "lc_observations": h_lc_obs.hexdigest()[:12],
            "lc_observation_tuples_ordered": h_lc_obs_order.hexdigest()[:12],
            "per_image_observation_counts": stable_tuple_json_hash(sorted(per_image_observations.items()))[:12],
        },
        "sample": sample,
    }


def pose_graph_summary(solve_state: Any) -> dict:
    items = sorted(solve_state.pair_records().items(), key=lambda item: int(item[0]))
    n_edges = 0
    n_valid = 0
    total_matches = 0
    sample = []
    h = hashlib.sha1()
    h_pair_id = hashlib.sha1()
    h_cfc_t = hashlib.sha1()
    h_cfc_q = hashlib.sha1()
    h_num_matches = hashlib.sha1()
    h_is_valid = hashlib.sha1()

    for pid, pair in items:
        n_edges += 1
        valid = bool(pair.is_valid)
        n_valid += int(valid)
        num_matches = int(np.asarray(pair.all_matches).shape[0])
        total_matches += num_matches
        pose = pair_cam2_from_cam1(pair)
        h.update(int(pid).to_bytes(8, "little", signed=False))
        h.update(b"\x01" if valid else b"\x00")
        h.update(int(num_matches).to_bytes(4, "little", signed=False))
        translation = None if pose is None else pose.translation
        h.update(hash_array_like(translation).encode())
        h.update(hash_array_like(rotation_quaternion(pose_rotation(pose))).encode())

        h_pair_id.update(int(pid).to_bytes(8, "little", signed=False))
        h_num_matches.update(int(num_matches).to_bytes(4, "little", signed=False))
        h_is_valid.update(b"\x01" if valid else b"\x00")
        h_cfc_t.update(hash_array_like(translation).encode())
        h_cfc_q.update(hash_array_like(rotation_quaternion(pose_rotation(pose))).encode())

        if len(sample) < 3:
            sample.append({"pair_id": int(pid), "is_valid": valid, "num_matches": num_matches})
    return {
        "n_edges": n_edges,
        "n_valid": n_valid,
        "total_matches": total_matches,
        "content_hash": h.hexdigest()[:16],
        "per_field_hash": {
            "pair_id": h_pair_id.hexdigest()[:12],
            "is_valid": h_is_valid.hexdigest()[:12],
            "num_matches": h_num_matches.hexdigest()[:12],
            "cam2_from_cam1.translation": h_cfc_t.hexdigest()[:12],
            "cam2_from_cam1.rotation.quat": h_cfc_q.hexdigest()[:12],
        },
        "sample": sample,
    }


def reconstruction_summary_from_tracks_summary(images: dict, track_fp: dict, cameras=None) -> dict:
    image_fp = images_summary(images)
    h = hashlib.sha1()
    h.update(image_fp["content_hash"].encode())
    h.update(track_fp["content_hash"].encode())
    summary = {
        "n_images": image_fp["n_images"],
        "n_registered": image_fp["n_registered"],
        "n_tracks": track_fp["n_tracks"],
        "total_observations": track_fp["total_observations"],
        "total_lc_observations": track_fp["total_lc_observations"],
        "content_hash": h.hexdigest()[:16],
        "images_per_field_hash": image_fp["per_field_hash"],
        "tracks_per_field_hash": track_fp["per_field_hash"],
    }
    if cameras is not None:
        summary["cameras"] = cameras_summary(cameras)
    return summary


def reconstruction_summary(images: dict, tracks: dict | None, cameras: dict | None = None) -> dict:
    return reconstruction_summary_from_tracks_summary(images, track_records_summary(tracks), cameras)


def colmap_reconstruction_summary(rec: Any) -> dict:
    if rec is None:
        return {"exists": False}
    if not all(hasattr(rec, name) for name in ("num_images", "num_reg_images", "num_points3D")):
        return {
            "exists": True,
            "num_images": len(getattr(rec, "images", {}) or {}),
            "num_reg_images": len(getattr(rec, "images", {}) or {}),
            "num_points3D": len(getattr(rec, "points3D", {}) or {}),
        }
    return {
        "exists": True,
        "num_images": int(rec.num_images()),
        "num_reg_images": int(rec.num_reg_images()),
        "num_points3D": int(rec.num_points3D()),
    }

"""Stage summaries and state captures for mapper byte evidence."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .canonical import (
    EVIDENCE_SCHEMA_VERSION,
    RELATIVE_POSE_STATE_SCHEMA_VERSION,
    array_hash,
    count_content_hash,
    file_hash,
    hash_array_like,
    image_pose,
    inlier_array,
    mapping_array_hash,
    pose_rotation,
    rotation_artifact,
    rotation_artifact_summary,
    rotation_quaternion,
)
from .fingerprints import fingerprint_scene
from .scene import (
    cameras_summary,
    colmap_reconstruction_summary,
    images_summary,
    native_problem_summary,
    ordered_observation_pairs,
    pose_graph_summary,
    reconstruction_summary,
    reconstruction_summary_from_tracks_summary,
    track_identity_map,
    track_identity_summary,
    track_observation_lists,
    track_records_summary,
)


def _track_state(tracks: dict | None) -> dict:
    state = {}
    for point3D_id, track in sorted((tracks or {}).items(), key=lambda item: int(item[0])):
        obs_list, lc_list = track_observation_lists(track)
        state[int(point3D_id)] = {
            "observations": ordered_observation_pairs(obs_list),
            "lc_observations": ordered_observation_pairs(lc_list),
            "xyz": np.asarray(track.xyz, dtype=np.float64),
            "is_initialized": False,
        }
    return state


def _image_mask_state(images: dict | None) -> dict:
    state = {}
    for image_id, image in sorted((images or {}).items(), key=lambda item: (0, int(item[0]))):
        state[int(image_id)] = {
            "is_depth_outlier": np.asarray(image.is_depth_outlier, dtype=bool),
            "is_track_anchor": np.asarray(image.is_track_anchor, dtype=bool),
            "is_inlier": np.asarray(image.is_inlier, dtype=bool),
            "is_excluded": np.asarray([], dtype=bool),
        }
    return state


def _pair_state(solve_state) -> dict:
    state = {}
    for pair_id, pair in sorted(
        solve_state.pair_records().items(),
        key=lambda item: int(item[0]),
    ):
        state[int(pair_id)] = {
            "is_valid": bool(pair.is_valid),
            "inliers": inlier_array(pair),
            "are_lc": np.asarray(pair.are_loop_closure, dtype=bool),
        }
    return state


def _rotation_summary(images: dict) -> dict:
    items = sorted((images or {}).items(), key=lambda item: int(item[0]))
    h = hashlib.sha1()
    active = 0
    sample = []
    for image_id, image in items:
        active += 1
        pose = image_pose(image)
        quat = rotation_quaternion(pose_rotation(pose)) if pose is not None else None
        h.update(int(image_id).to_bytes(8, "little", signed=False))
        h.update(hash_array_like(quat).encode())
        if len(sample) < 3:
            sample.append({"image_id": int(image_id)})
    return {
        "active_image_count": active,
        "rotation_fingerprint_hash": h.hexdigest()[:16],
        "sample": sample,
    }


def _rotation_averaging_summary(
    solve_state,
    images: dict,
    *,
    success: bool | None = None,
    weights: dict | None = None,
    filtered_consecutive_pairs=None,
    random_seed=None,
    fixed_image_id=None,
) -> dict:
    graph = native_problem_summary(solve_state)
    image_fp = images_summary(images)
    rotation_fp = _rotation_summary(images)
    rotation_summary = rotation_artifact_summary(images)
    active_image_ids = sorted(int(iid) for iid in (images or {}).keys())
    inferred_fixed_image_id = (
        int(fixed_image_id) if fixed_image_id is not None else (active_image_ids[0] if active_image_ids else None)
    )
    return {
        "graph": graph,
        "images": image_fp,
        "active_image_count": rotation_fp["active_image_count"],
        "valid_pair_count": graph["n_valid"],
        "fixed_gauge_image_id": inferred_fixed_image_id,
        "fixed_gauge_image_source": ("explicit" if fixed_image_id is not None else "first_active_image"),
        "random_initialization": False,
        "random_initialization_evidence": (
            "base RA calls pyglomap.run_rotation_averaging directly with opt_ra; "
            "only GP has random_init_scale/use_init state in this code path"
        ),
        "random_seed": random_seed,
        "rotations": rotation_fp,
        "rotation_artifact": rotation_summary,
        "success": None if success is None else bool(success),
        "weights": {
            "num_weights": len(weights or {}),
            "hash": hash_array_like([(int(pid), float(weight)) for pid, weight in sorted((weights or {}).items())]),
        },
        "filtered_consecutive_pairs": sorted(int(pid) for pid in (filtered_consecutive_pairs or set())),
    }


def _scale_stats(scales: dict | None) -> dict:
    if not scales:
        return {"count": 0}
    values = np.asarray(list(scales.values()), dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"count": int(len(values)), "finite": 0}
    return {
        "count": int(len(values)),
        "finite": int(len(finite)),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "hash": array_hash(values)[:12],
    }


def gp_input_rotations(images) -> dict:
    return rotation_artifact(images)


def database_file_summary(path: Path, *, include_file_hash: bool = True) -> dict:
    summary = {
        "path_name": path.name,
        "size": path.stat().st_size,
    }
    if include_file_hash:
        summary["sha256"] = file_hash(path)
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True) as db:
        available_tables = {str(name) for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        tables = {}
        for name in (
            "cameras",
            "images",
            "keypoints",
            "matches",
            "two_view_geometries",
        ):
            if name in available_tables:
                tables[name] = int(db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            else:
                tables[name] = None
        summary["tables"] = tables
    return summary


def database_to_native_summary(database_path: Path, solve_state, images) -> dict:
    return {
        "database": database_file_summary(Path(database_path), include_file_hash=False),
        "graph": native_problem_summary(solve_state),
        "images": images_summary(images),
    }


def relative_pose_summary(
    solve_state,
    images,
    filtered_consecutive_pairs,
    mdrp_results,
    mdrp_depth_outlier_masks,
) -> dict:
    return {
        "graph": native_problem_summary(solve_state),
        "images": images_summary(images),
        "filtered_consecutive_pairs": sorted(int(pid) for pid in (filtered_consecutive_pairs or set())),
        "num_mdrp_results": len(mdrp_results or {}),
        "num_mdrp_depth_masks": len(mdrp_depth_outlier_masks or {}),
    }


def capture_relative_pose_state(
    solve_state,
    images,
    consecutive_pair_ids,
    filtered_consecutive_pairs,
    mdrp_results,
    mdrp_depth_outlier_masks,
) -> dict:
    pair_state = {}
    consecutive_pair_ids = set(consecutive_pair_ids or [])
    mdrp_results = mdrp_results or {}
    for pid, pair in solve_state.pair_records().items():
        geom = pair.geometry
        result = mdrp_results.get(pid, mdrp_results.get(int(pid)))
        if isinstance(result, Mapping):
            rel_depth_scale = float(result.get("rel_depth_scale", 1.0))
            weight = float(result.get("weight", 0.0))
            if int(pid) in consecutive_pair_ids:
                weight = 138.0
        else:
            rel_depth_scale = -1.0
            weight = 0.0
        pair_state[int(pid)] = {
            "is_valid": bool(pair.is_valid),
            "cam2_from_cam1": solve_state.export_pair_pose(pid),
            "inliers": inlier_array(pair),
            "weight": weight,
            "rel_depth_scale": rel_depth_scale,
            "config": int(geom.configuration),
        }

    image_state = {}
    for iid, image in sorted(images.items(), key=lambda item: int(item[0])):
        image_state[int(iid)] = {
            "depth_priors": np.asarray(image.depth_values),
            "depth_prior_stddevs": np.asarray(image.depth_stddevs),
            "depth_prior_validity": np.asarray(image.depth_validity, dtype=bool),
        }

    return {
        "schema": "videosfm.base_cache_record.relative_pose_state",
        "schema_version": RELATIVE_POSE_STATE_SCHEMA_VERSION,
        "pair_state": pair_state,
        "image_state": image_state,
        "filtered_consecutive_pairs": set(filtered_consecutive_pairs or set()),
        "mdrp_depth_outlier_masks": mdrp_depth_outlier_masks,
        "mdrp_results_cache": mdrp_results,
        "summary": relative_pose_summary(
            solve_state,
            images,
            filtered_consecutive_pairs,
            mdrp_results,
            mdrp_depth_outlier_masks,
        ),
    }


def ra_summary(solve_state, images, filtered_consecutive_pairs) -> dict:
    summary = _rotation_averaging_summary(
        solve_state,
        images,
        success=True,
        weights=None,
        filtered_consecutive_pairs=filtered_consecutive_pairs,
    )
    summary["rotation_artifact"] = {
        **summary["rotation_artifact"],
        "file": "rotations.json",
    }
    return summary


def tracks_summary(
    solve_state,
    images,
    tracks_full,
    tracks,
    mdrp_depth_outlier_masks,
    boundary_depth_outliers_marked,
) -> dict:
    return {
        "graph": native_problem_summary(solve_state),
        "images": images_summary(images),
        "tracks_full": track_records_summary(tracks_full or {}),
        "tracks": track_records_summary(tracks or {}),
        "num_track_anchor_masks": 0,
        "num_mdrp_depth_masks": len(mdrp_depth_outlier_masks or {}),
        "boundary_depth_outliers_marked": bool(boundary_depth_outliers_marked),
    }


def ba_start_summary(rec, solve_state, tracks=None, *, track_summary=None) -> dict:
    """Structural live snapshot at bundle-adjustment entry."""

    if track_summary is None:
        track_summary = track_records_summary(tracks or {})
    images = solve_state.image_records()

    return {
        "stage": "ba_start",
        "reconstruction": {
            "colmap": colmap_reconstruction_summary(rec),
            "native": reconstruction_summary_from_tracks_summary(
                images,
                track_summary,
                rec.cameras,
            ),
            "cameras": cameras_summary(rec.cameras),
        },
        "view_graph": native_problem_summary(solve_state),
        "images": images_summary(images),
        "tracks": track_summary,
        "summary_scope": "reconstruction_state_only",
    }


def capture_tracks_state(
    solve_state,
    images,
    tracks_full,
    tracks,
    mdrp_depth_outlier_masks,
    boundary_depth_outliers_marked,
) -> dict:
    return {
        "schema": "videosfm.base_cache_record.tracks_state",
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "tracks_full": _track_state(tracks_full),
        "tracks": _track_state(tracks),
        "image_masks": _image_mask_state(images),
        "pair_state": _pair_state(solve_state),
        "track_anchor_masks": {},
        "mdrp_depth_outlier_masks": {
            int(image_id): np.asarray(mask) for image_id, mask in (mdrp_depth_outlier_masks or {}).items()
        },
        "boundary_depth_outliers_marked": bool(boundary_depth_outliers_marked),
    }


def gp_input_summary(
    stage: str,
    solve_state,
    source_images,
    source_tracks,
    cameras,
) -> dict:
    summary = {
        "stage": stage,
        "pose_graph": pose_graph_summary(solve_state),
        "gp_rec": reconstruction_summary(source_images, source_tracks, cameras),
        "source_graph": native_problem_summary(solve_state),
        "source_images": images_summary(source_images),
        "source_tracks": track_records_summary(source_tracks),
        "rotation_artifact": {
            **rotation_artifact_summary(source_images),
            "file": "input_rotations.json",
        },
    }
    if stage == "gp2":
        _stabilize_gp2_input_summary(summary)
    if cameras is not None:
        summary["source_cameras"] = cameras_summary(cameras)
    return summary


def _stabilize_gp2_input_summary(summary: dict) -> None:
    marker = "omitted_gp2_pre_roundtrip"
    for key in ("gp_rec", "source_images"):
        section = summary.get(key)
        if not isinstance(section, dict):
            continue
        fields_key = (
            "images_per_field_hash" if isinstance(section.get("images_per_field_hash"), dict) else "per_field_hash"
        )
        fields = section.get(fields_key)
        if isinstance(fields, dict):
            fields["cam_from_world.translation"] = marker
            fields["solver_internal.center"] = marker
        section["content_hash"] = count_content_hash(
            "gp2_input_images",
            section.get("n_images"),
            section.get("n_registered"),
            section.get("total_features"),
            section.get("total_depth_priors"),
            section.get("n_tracks"),
            section.get("total_observations"),
        )


def _require_gp_debug_result(stage: str, result: dict) -> None:
    missing = [
        key
        for key in (
            "debug_initial_frame_centers",
            "debug_initial_point3D_xyz",
            "debug_initial_bata_scales",
        )
        if key not in result
    ]
    if missing:
        raise RuntimeError(
            f"Replay cache write for {stage} requires GP init debug outputs from vidmap_native; missing {missing}"
        )


def _gp_init_vector_map(value: Mapping | None) -> dict[int, list[float]]:
    if not value:
        return {}
    return {
        int(key): [float(x) for x in np.asarray(value[key], dtype=np.float64).reshape(3)]
        for key in sorted(value, key=int)
    }


def _gp_init_scalar_map(value: Mapping | None) -> dict:
    if not value:
        return {}
    return {key: float(value[key]) for key in sorted(value, key=str)}


def capture_gp_initial_state(
    stage: str,
    native_opts,
    result: dict,
    input_summary: dict,
    tracks: dict | None = None,
) -> dict:
    _require_gp_debug_result(stage, result)
    initial_dmap_scales = dict(native_opts.initial_depth_map_scales)
    if not initial_dmap_scales:
        initial_dmap_scales = {int(image_id): 1.0 for image_id in sorted(result["dmap_scale_map"])}
    return {
        "stage": stage,
        "initial_dmap_scales": initial_dmap_scales,
        "debug_initial_frame_centers": _gp_init_vector_map(result["debug_initial_frame_centers"]),
        "debug_initial_point3D_xyz": _gp_init_vector_map(result["debug_initial_point3D_xyz"]),
        "debug_initial_bata_scales": _gp_init_scalar_map(result["debug_initial_bata_scales"]),
        "track_identity_by_point3D_id": track_identity_map(tracks),
        "input_summary": input_summary,
    }


def gp_initial_state_summary(state: dict) -> dict:
    return {
        "stage": state["stage"],
        "initial_dmap_scales": _scale_stats(state["initial_dmap_scales"]),
        "frame_centers_hash": mapping_array_hash(state["debug_initial_frame_centers"]),
        "point3D_xyz_hash": mapping_array_hash(state["debug_initial_point3D_xyz"]),
        "bata_scales_hash": mapping_array_hash(state["debug_initial_bata_scales"]),
        "track_identity": track_identity_summary(state["track_identity_by_point3D_id"]),
        "input_summary": state["input_summary"],
    }


def gp_output_summary(stage: str, state, result: dict) -> dict:
    tracks = state.track_records()
    return {
        "stage": stage,
        "gp_rec": fingerprint_scene(
            state.image_records(),
            tracks,
            state.reconstruction.cameras,
        ),
        "dmap_scale_map": _scale_stats(dict(result["dmap_scale_map"])),
        "success": bool(result["success"]),
        "debug_diagnostics": dict(result["debug_diagnostics"]),
    }

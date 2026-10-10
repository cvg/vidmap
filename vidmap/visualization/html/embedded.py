"""Package one run inside the browser reconstruction viewer."""

from __future__ import annotations

import base64
import gzip
import io
import sqlite3
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from vidmap.mapper.inputs import LC_MASKS_NAME
from vidmap.utils.loop_closure_masks import read_loop_closure_masks

from . import scene
from .exporter import write_html

_PAIR_ID_BASE = 2147483647
_QUERY_CHUNK_SIZE = 500
_VALID_TWO_VIEW_CONFIGS = frozenset({2, 3, 4, 5, 6, 9})


def _embed_image_previews(
    images_dir: Path,
    image_names: tuple[str, ...],
    *,
    maximum_size: int = 640,
) -> tuple[str, ...]:
    """Encode bounded JPEG previews for an embedded keyframe timeline."""
    if maximum_size < 1:
        raise ValueError("preview maximum size must be at least one")
    resolved_images_dir = images_dir.resolve()
    previews = []
    for name in image_names:
        normalized_name = name.replace("\\", "/")
        if normalized_name.startswith("/") or ".." in normalized_name.split("/"):
            raise ValueError(f"Reconstruction image name is not a safe relative path: {name}")
        image_path = resolved_images_dir / normalized_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Timeline image does not exist: {image_path}")
        with Image.open(image_path) as source:
            preview = source.convert("RGB")
            preview.thumbnail((maximum_size, maximum_size), Image.Resampling.LANCZOS)
            encoded = io.BytesIO()
            preview.save(encoded, format="JPEG", quality=80, optimize=True)
        previews.append(f"data:image/jpeg;base64,{base64.b64encode(encoded.getvalue()).decode('ascii')}")
    return tuple(previews)


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Embedded viewer requires {label}: {path}")
    return path


def _encoded_bytes(data: bytes) -> dict[str, object]:
    compressed = gzip.compress(data, compresslevel=6, mtime=0)
    return {
        "encoding": "gzip-base64",
        "size": len(data),
        "base64": base64.b64encode(compressed).decode("ascii"),
    }


def _encoded_file(path: Path) -> dict[str, object]:
    return _encoded_bytes(path.read_bytes())


def _pair_id(first: int, second: int) -> int:
    return min(first, second) * _PAIR_ID_BASE + max(first, second)


def _rows_for_pair_ids(connection, table: str, columns: str, pair_ids: tuple[int, ...]):
    for start in range(0, len(pair_ids), _QUERY_CHUNK_SIZE):
        chunk = pair_ids[start : start + _QUERY_CHUNK_SIZE]
        placeholders = ",".join("?" for _ in chunk)
        yield from connection.execute(
            f"SELECT {columns} FROM {table} WHERE pair_id IN ({placeholders})",  # noqa: S608
            chunk,
        )


def _compact_loop_closure_database(
    database: Path,
    loop_closure_masks: Mapping[tuple[str, str], np.ndarray],
) -> bytes:
    """Retain only the SQLite rows queried by the embedded LC viewer."""
    uri = f"{database.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as source:
        image_ids = {
            str(name): int(image_id) for image_id, name in source.execute("SELECT image_id, name FROM images")
        }
        selected_names: set[str] = set()
        pair_ids = []
        for (first, second), mask in loop_closure_masks.items():
            if not mask.any():
                continue
            if first not in image_ids or second not in image_ids:
                continue
            selected_names.update((first, second))
            pair_ids.append(_pair_id(image_ids[first], image_ids[second]))
        ordered_pair_ids = tuple(sorted(set(pair_ids)))
        geometry_rows = tuple(
            row
            for row in _rows_for_pair_ids(
                source,
                "two_view_geometries",
                "pair_id, rows, config",
                ordered_pair_ids,
            )
            if int(row[1]) > 0 and row[2] is not None and int(row[2]) in _VALID_TWO_VIEW_CONFIGS
        )
        valid_pair_ids = tuple(sorted(int(row[0]) for row in geometry_rows))
        match_rows = tuple(_rows_for_pair_ids(source, "matches", "pair_id, rows, cols, data", valid_pair_ids))
        selected_images = tuple(sorted((image_ids[name], name) for name in selected_names))

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as stream:
            temporary = Path(stream.name)
        with sqlite3.connect(temporary) as target:
            target.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, config INTEGER);
                CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
                """
            )
            target.executemany("INSERT INTO images VALUES (?, ?)", selected_images)
            target.executemany("INSERT INTO two_view_geometries VALUES (?, ?, ?)", geometry_rows)
            target.executemany("INSERT INTO matches VALUES (?, ?, ?, ?)", match_rows)
        return temporary.read_bytes()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _image_previews(
    run: Path,
    images_dir: Path | None,
    reconstruction_dir: Path | None = None,
) -> dict[str, dict[str, object]]:
    reconstruction = reconstruction_dir if reconstruction_dir is not None else run / "rec"
    if images_dir is None:
        from vidmap.reconstruction import local_run_image_dir

        images_dir = local_run_image_dir(run)
        if images_dir is None:
            return {}
    images_dir = images_dir.expanduser().resolve(strict=True)

    import pycolmap

    model = pycolmap.Reconstruction(reconstruction)
    image_by_name = {str(image.name): image for image in model.images.values() if image.has_pose}
    names = tuple(sorted(image_by_name))
    previews = _embed_image_previews(images_dir, names)
    return {
        name: {
            "url": preview,
            "width": int(model.cameras[image_by_name[name].camera_id].width),
            "height": int(model.cameras[image_by_name[name].camera_id].height),
        }
        for name, preview in zip(names, previews, strict=True)
    }


def _camera_frustum_segments(image, camera) -> tuple[list[float], list[float]]:
    center = np.asarray(image.projection_center(), dtype=np.float64)
    width = float(camera.width)
    height = float(camera.height)
    fx = float(camera.focal_length_x)
    fy = float(camera.focal_length_y)
    cx = float(camera.principal_point_x)
    cy = float(camera.principal_point_y)
    image_extent = max(0.3 * width / 1024.0, 0.3 * height / 1024.0)
    world_extent = 2.0 * max(width, height) / (fx + fy)
    scale = 0.5 * image_extent / world_extent
    rot_cw = np.asarray(image.cam_from_world().rotation.matrix(), dtype=np.float64)
    pixel_corners = ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    corners = []
    for u, v in pixel_corners:
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        corners.append(center + 0.5 * scale * (rot_cw.T @ ray))
    segments: list[float] = []
    for corner in corners:
        segments.extend(round(float(x), 5) for x in center)
        segments.extend(round(float(x), 5) for x in corner)
    for idx in range(4):
        segments.extend(round(float(x), 5) for x in corners[idx])
        segments.extend(round(float(x), 5) for x in corners[(idx + 1) % 4])
    return [round(float(x), 5) for x in center], segments


def _compute_sim3_est_from_gt(
    est_rec,
    gt_rec,
):
    import pycolmap

    from vidmap.benchmark.trajectory import align_umeyama_sim3

    if est_rec.num_reg_images() < 2 or gt_rec.num_reg_images() < 2:
        return None

    est_posed = sorted(
        (img for img in est_rec.images.values() if img.has_pose),
        key=lambda img: str(img.name),
    )
    est_by_name = {str(img.name): img for img in est_posed}
    min_est_name = str(est_posed[0].name)
    max_est_name = str(est_posed[-1].name)

    gt_all_posed = sorted(
        (img for img in gt_rec.images.values() if img.has_pose),
        key=lambda img: str(img.name),
    )
    gt_in_span = [img for img in gt_all_posed if min_est_name <= str(img.name) <= max_est_name]
    if len(gt_in_span) < 2:
        gt_in_span = gt_all_posed
    gt_by_name = {str(img.name): img for img in gt_in_span}

    common_names = sorted(set(est_by_name) & set(gt_by_name))
    if len(common_names) >= 2:
        p_es = np.array([est_by_name[n].projection_center() for n in common_names], dtype=np.float64)
        p_gt = np.array([gt_by_name[n].projection_center() for n in common_names], dtype=np.float64)
    else:
        from vidmap.utils.trajectory import remap_poses_to_timeline

        mapped_est = remap_poses_to_timeline(gt_rec, est_rec).reconstruction
        mapped_names = sorted(
            str(img.name) for img in mapped_est.images.values() if img.has_pose and str(img.name) in gt_by_name
        )
        if len(mapped_names) < 2:
            return None
        mapped_by_name = {str(img.name): img for img in mapped_est.images.values() if img.has_pose}
        common_names = mapped_names
        p_es = np.array([mapped_by_name[n].projection_center() for n in common_names], dtype=np.float64)
        p_gt = np.array([gt_by_name[n].projection_center() for n in common_names], dtype=np.float64)

    if len(common_names) >= 3:
        s_u, rot_u, trans_u = align_umeyama_sim3(p_es, p_gt)
        if not np.isfinite(s_u) or s_u <= 1e-6:
            return None
        sim3_gt_from_est = pycolmap.Sim3d(
            scale=float(s_u),
            rotation=pycolmap.Rotation3d(rot_u),
            translation=np.asarray(trans_u, dtype=np.float64),
        )

        def _med_err(sim3: pycolmap.Sim3d) -> float:
            rot_m = np.asarray(sim3.rotation.matrix(), dtype=np.float64)
            t_v = np.asarray(sim3.translation, dtype=np.float64)
            aligned = float(sim3.scale) * (p_es @ rot_m.T) + t_v
            return float(np.median(np.linalg.norm(p_gt - aligned, axis=1)))

        est_tmp = pycolmap.Reconstruction()
        gt_tmp = pycolmap.Reconstruction()
        for idx_n, n in enumerate(common_names, start=1):
            e_im = est_by_name.get(n)
            g_im = gt_by_name[n]
            if e_im is None:
                continue
            if e_im.camera_id not in est_tmp.cameras:
                est_tmp.add_camera_with_trivial_rig(est_rec.cameras[e_im.camera_id])
            if g_im.camera_id not in gt_tmp.cameras:
                gt_tmp.add_camera_with_trivial_rig(gt_rec.cameras[g_im.camera_id])
            est_tmp.add_image_with_trivial_frame(
                pycolmap.Image(image_id=idx_n, camera_id=e_im.camera_id, name=n), e_im.cam_from_world()
            )
            gt_tmp.add_image_with_trivial_frame(
                pycolmap.Image(image_id=idx_n, camera_id=g_im.camera_id, name=n), g_im.cam_from_world()
            )

        best_med = _med_err(sim3_gt_from_est)
        if est_tmp.num_reg_images() >= 3:
            for max_error in (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0):
                cand = pycolmap.align_reconstructions_via_proj_centers(
                    est_tmp,
                    gt_tmp,
                    max_proj_center_error=max_error,
                )
                if cand is not None and np.isfinite(cand.scale) and cand.scale > 1e-6:
                    cand_med = _med_err(cand)
                    if cand_med < best_med:
                        best_med = cand_med
                        sim3_gt_from_est = cand
    else:
        dist_es = float(np.linalg.norm(p_es[1] - p_es[0]))
        dist_gt = float(np.linalg.norm(p_gt[1] - p_gt[0]))
        s_2 = dist_gt / max(1e-6, dist_es) if dist_es > 1e-6 else 1.0
        m_sum = np.zeros((3, 3), dtype=np.float64)
        for n in common_names:
            r_es = np.asarray(est_by_name[n].cam_from_world().rotation.matrix(), dtype=np.float64)
            r_gt = np.asarray(gt_by_name[n].cam_from_world().rotation.matrix(), dtype=np.float64)
            m_sum += r_gt.T @ r_es
        u_m, _, vt_m = np.linalg.svd(m_sum)
        s_diag = np.eye(3)
        if np.linalg.det(u_m) * np.linalg.det(vt_m) < 0:
            s_diag[2, 2] = -1.0
        rot_2 = u_m @ s_diag @ vt_m
        trans_2 = p_gt.mean(axis=0) - s_2 * (rot_2 @ p_es.mean(axis=0))
        sim3_gt_from_est = pycolmap.Sim3d(
            scale=float(s_2),
            rotation=pycolmap.Rotation3d(rot_2),
            translation=np.asarray(trans_2, dtype=np.float64),
        )

    return sim3_gt_from_est.inverse()


def _covariance_ellipsoid_segments(
    center: np.ndarray,
    cov_3x3: np.ndarray,
    num_ring_segments: int = 24,
) -> list[float]:
    eigvals, eigvecs = np.linalg.eigh(cov_3x3)
    radii = np.sqrt(np.maximum(eigvals, 1e-12))
    angles = np.linspace(0.0, 2.0 * np.pi, num_ring_segments + 1, dtype=np.float64)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    segments: list[float] = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        va = eigvecs[:, a] * radii[a]
        vb = eigvecs[:, b] * radii[b]
        ring_pts = center[None, :] + cos_a[:, None] * va[None, :] + sin_a[:, None] * vb[None, :]
        for s in range(num_ring_segments):
            segments.extend(round(float(x), 5) for x in ring_pts[s])
            segments.extend(round(float(x), 5) for x in ring_pts[s + 1])
    return segments


def _trajectory_payload_from_sim3(
    est_rec,
    gt_reconstruction_dir: Path,
    sim3_est_from_gt,
    *,
    frusta_at_est_keyframes_only: bool = False,
) -> dict[str, object] | None:
    import bisect
    import json

    import pycolmap

    if est_rec.num_reg_images() < 2 or sim3_est_from_gt is None:
        return None

    est_posed = sorted(
        (img for img in est_rec.images.values() if img.has_pose),
        key=lambda img: str(img.name),
    )
    est_by_name = {str(img.name): img for img in est_posed}
    min_est_name = str(est_posed[0].name)
    max_est_name = str(est_posed[-1].name)

    gt_in_est = pycolmap.Reconstruction(gt_reconstruction_dir)
    if gt_in_est.num_reg_images() < 2:
        return None
    gt_in_est.transform(sim3_est_from_gt)

    gt_all_posed = sorted(
        (img for img in gt_in_est.images.values() if img.has_pose),
        key=lambda img: str(img.name),
    )
    gt_in_span = [img for img in gt_all_posed if min_est_name <= str(img.name) <= max_est_name]
    gt_posed = gt_in_span if len(gt_in_span) >= 2 else gt_all_posed

    gt_names = [str(img.name) for img in gt_posed]
    centers: list[float] = []
    for img in gt_posed:
        centers.extend(round(float(x), 5) for x in img.projection_center())

    if frusta_at_est_keyframes_only and len(gt_posed) > len(est_posed):
        frusta_imgs = [img for img in gt_posed if str(img.name) in est_by_name]
        if len(frusta_imgs) < 2:
            step = max(1, len(gt_posed) // max(1, len(est_posed)))
            frusta_imgs = gt_posed[::step]
    else:
        frusta_imgs = gt_posed

    # Optional per-image metadata of the reference poses, next to the reference reconstruction:
    #   pose_covariances.npz: image_names (N,), cov_position (N, 3, 3) camera-center covariance in world units,
    #                         confidence (N,), num_inliers (N,), reproj_rms_px (N,)
    #   pose_confidence.json: {image_name: confidence} (used if pose_covariances.npz is absent)
    cov_npz_path = gt_reconstruction_dir / "pose_covariances.npz"
    conf_json_path = gt_reconstruction_dir / "pose_confidence.json"
    cov_info_by_name: dict[str, tuple[np.ndarray, float, int, float]] = {}
    conf_only_by_name: dict[str, float] = {}
    if cov_npz_path.is_file():
        cov_data = np.load(cov_npz_path)
        c_names = [str(x) for x in cov_data["image_names"]]
        c_pos = np.asarray(cov_data["cov_position"], dtype=np.float64)
        c_conf = np.asarray(cov_data["confidence"], dtype=np.float64)
        c_inl = np.asarray(cov_data["num_inliers"], dtype=np.int32)
        c_rms = np.asarray(cov_data["reproj_rms_px"], dtype=np.float64)
        for i, n in enumerate(c_names):
            cov_info_by_name[n] = (c_pos[i], float(c_conf[i]), int(c_inl[i]), float(c_rms[i]))
    elif conf_json_path.is_file():
        conf_only_by_name = {
            str(k): float(v) for k, v in json.loads(conf_json_path.read_text(encoding="utf-8")).items()
        }

    scale_est_from_gt = float(sim3_est_from_gt.scale)
    rot_est_from_gt = np.asarray(sim3_est_from_gt.rotation.matrix(), dtype=np.float64)

    frusta_names = [str(img.name) for img in frusta_imgs]
    keyframe_centers: list[float] = []
    keyframe_frusta: list[float] = []
    keyframe_cov_ellipses: list[float] = []
    keyframe_confidences: list[float] = []
    keyframe_inliers: list[int] = []
    keyframe_pos_std_meters: list[list[float]] = []
    keyframe_reproj_rms_px: list[float] = []

    has_cov = bool(cov_info_by_name)
    has_conf = bool(cov_info_by_name or conf_only_by_name)

    for img in frusta_imgs:
        name = str(img.name)
        cam = gt_in_est.cameras[img.camera_id]
        kf_center, kf_frustum = _camera_frustum_segments(img, cam)
        keyframe_centers.extend(kf_center)
        keyframe_frusta.extend(kf_frustum)
        if has_cov and name in cov_info_by_name:
            cov_world, conf_val, inl_val, rms_val = cov_info_by_name[name]
            cov_viewer = (scale_est_from_gt**2) * (rot_est_from_gt @ cov_world @ rot_est_from_gt.T)
            center_vec = np.asarray(img.projection_center(), dtype=np.float64)
            keyframe_cov_ellipses.extend(_covariance_ellipsoid_segments(center_vec, cov_viewer))
            keyframe_confidences.append(round(conf_val, 5))
            keyframe_inliers.append(inl_val)
            evals_m = np.maximum(np.linalg.eigvalsh(cov_world), 0.0)
            axes_m = np.sqrt(evals_m)[::-1]
            std_tot_m = float(np.sqrt(np.sum(evals_m)))
            keyframe_pos_std_meters.append(
                [
                    round(std_tot_m, 4),
                    round(float(axes_m[0]), 4),
                    round(float(axes_m[1]), 4),
                    round(float(axes_m[2]), 4),
                ]
            )
            keyframe_reproj_rms_px.append(round(rms_val, 3) if np.isfinite(rms_val) else 0.0)
        elif has_cov:
            center_vec = np.asarray(img.projection_center(), dtype=np.float64)
            keyframe_cov_ellipses.extend(_covariance_ellipsoid_segments(center_vec, np.eye(3) * 1e-6))
            keyframe_confidences.append(round(conf_only_by_name.get(name, 1.0), 5))
            keyframe_inliers.append(0)
            keyframe_pos_std_meters.append([0.0, 0.0, 0.0, 0.0])
            keyframe_reproj_rms_px.append(0.0)
        elif has_conf:
            keyframe_confidences.append(round(conf_only_by_name.get(name, 1.0), 5))

    keyframe_path_counts: list[int] = []
    keyframe_frusta_counts: list[int] = []
    for idx, est_img in enumerate(est_posed):
        name = str(est_img.name)
        if idx == len(est_posed) - 1:
            keyframe_path_counts.append(len(gt_names))
            keyframe_frusta_counts.append(len(frusta_names))
        else:
            keyframe_path_counts.append(bisect.bisect_right(gt_names, name))
            keyframe_frusta_counts.append(bisect.bisect_right(frusta_names, name))

    result: dict[str, object] = {
        "centers": centers,
        "keyframePathCounts": keyframe_path_counts,
        "keyframeFrustaCounts": keyframe_frusta_counts,
        "keyframeCenters": keyframe_centers,
        "keyframeFrusta": keyframe_frusta,
        "keyframeNames": frusta_names,
    }
    if has_conf:
        result["keyframeConfidences"] = keyframe_confidences
    if has_cov:
        result["keyframeCovEllipses"] = keyframe_cov_ellipses
        result["keyframeInliers"] = keyframe_inliers
        result["keyframePosStdMeters"] = keyframe_pos_std_meters
        result["keyframeReprojRmsPx"] = keyframe_reproj_rms_px
    return result


def _embedded_run_payload(
    run_dir: str | Path,
    *,
    reconstruction_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    gt_reconstruction_dir: str | Path | None = None,
    dense_gt_reconstruction_dir: str | Path | None = None,
) -> dict[str, object]:
    """Package one normalized run for the browser's ordinary load path."""
    run = Path(run_dir).expanduser().resolve(strict=True)
    reconstruction = (
        Path(reconstruction_dir).expanduser().resolve(strict=True) if reconstruction_dir is not None else run / "rec"
    )
    mapper_inputs = run / "mapper_inputs"
    source_files = {
        f"rec/{name}": _required_file(reconstruction / name, f"rec/{name}")
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    }
    files = {name: _encoded_file(path) for name, path in source_files.items()}
    database = mapper_inputs / "database_complete.db"
    loop_closure_masks_path = mapper_inputs / LC_MASKS_NAME
    if database.is_file() != loop_closure_masks_path.is_file():
        raise FileNotFoundError(
            "Embedded loop-closure inspection requires both " f"{database} and {loop_closure_masks_path}"
        )
    if database.is_file():
        loop_closure_masks = read_loop_closure_masks(loop_closure_masks_path)
        files["mapper_inputs/database_complete.db"] = _encoded_bytes(
            _compact_loop_closure_database(database, loop_closure_masks)
        )
        files[f"mapper_inputs/{LC_MASKS_NAME}"] = _encoded_file(loop_closure_masks_path)
    covariance = reconstruction / "visualization_cache" / "point_covariance_rank_v2.bin"
    if covariance.is_file():
        files["rec/visualization_cache/point_covariance_rank_v2.bin"] = _encoded_file(covariance)
    payload: dict[str, object] = {
        "files": files,
        "imagePreviews": _image_previews(
            run,
            None if images_dir is None else Path(images_dir),
            reconstruction_dir=reconstruction,
        ),
    }
    gt_dir = (
        Path(gt_reconstruction_dir).expanduser().resolve(strict=True)
        if gt_reconstruction_dir is not None
        else (run / "gt_rec" if (run / "gt_rec" / "images.bin").is_file() else None)
    )
    dense_gt_dir = (
        Path(dense_gt_reconstruction_dir).expanduser().resolve(strict=True)
        if dense_gt_reconstruction_dir is not None
        else (run / "dense_gt_rec" if (run / "dense_gt_rec" / "images.bin").is_file() else None)
    )
    if gt_dir is not None or dense_gt_dir is not None:
        import pycolmap

        est_rec = pycolmap.Reconstruction(reconstruction)
        sim3_est_from_gt = None
        if gt_dir is not None:
            gt_rec = pycolmap.Reconstruction(gt_dir)
            sim3_est_from_gt = _compute_sim3_est_from_gt(est_rec, gt_rec)
            gt_trajectory = _trajectory_payload_from_sim3(
                est_rec,
                gt_dir,
                sim3_est_from_gt,
                frusta_at_est_keyframes_only=(dense_gt_dir is None),
            )
            if gt_trajectory is not None:
                payload["gtTrajectory"] = gt_trajectory
        if dense_gt_dir is not None:
            dense_sim3 = sim3_est_from_gt
            if dense_sim3 is None:
                dense_rec = pycolmap.Reconstruction(dense_gt_dir)
                dense_sim3 = _compute_sim3_est_from_gt(est_rec, dense_rec)
            dense_trajectory = _trajectory_payload_from_sim3(
                est_rec,
                dense_gt_dir,
                dense_sim3,
                frusta_at_est_keyframes_only=True,
            )
            if dense_trajectory is not None:
                payload["denseGtTrajectory"] = dense_trajectory
    return payload


def write_embedded_viewer_html(
    run_dir: str | Path,
    output: str | Path,
    *,
    reconstruction_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    gt_reconstruction_dir: str | Path | None = None,
    dense_gt_reconstruction_dir: str | Path | None = None,
) -> Path:
    """Write an embedded viewer that automatically loads one run or sub-reconstruction."""
    payload = _embedded_run_payload(
        run_dir,
        reconstruction_dir=reconstruction_dir,
        images_dir=images_dir,
        gt_reconstruction_dir=gt_reconstruction_dir,
        dense_gt_reconstruction_dir=dense_gt_reconstruction_dir,
    )
    return write_html(output, scene.render_viewer_html(embedded_run=payload))


def write_all_embedded_viewers(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    images_dir: str | Path | None = None,
    base_name: str | None = None,
    gt_reconstruction_dir: str | Path | None = None,
    dense_gt_reconstruction_dir: str | Path | None = None,
) -> list[Path]:
    """Write embedded viewer HTMLs for all reconstructions or sub-reconstructions in a run."""
    run = Path(run_dir).expanduser().resolve(strict=True)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else run
    out_dir.mkdir(parents=True, exist_ok=True)

    rec_dir = run / "rec"
    sub_dirs = (
        sorted([d for d in rec_dir.iterdir() if d.is_dir() and (d / "images.bin").is_file()])
        if rec_dir.is_dir()
        else []
    )

    written = []
    if sub_dirs:
        for sub_dir in sub_dirs:
            stem = base_name if base_name is not None else "vidmap-viewer-embedded"
            sep = "-" if stem == "vidmap-viewer-embedded" else "_"
            out_file = out_dir / f"{stem}{sep}sub{sub_dir.name}.html"
            write_embedded_viewer_html(
                run,
                out_file,
                reconstruction_dir=sub_dir,
                images_dir=images_dir,
                gt_reconstruction_dir=gt_reconstruction_dir,
                dense_gt_reconstruction_dir=dense_gt_reconstruction_dir,
            )
            written.append(out_file)
    else:
        stem = base_name if base_name is not None else "vidmap-viewer-embedded"
        out_file = out_dir / f"{stem}.html"
        write_embedded_viewer_html(
            run,
            out_file,
            reconstruction_dir=rec_dir,
            images_dir=images_dir,
            gt_reconstruction_dir=gt_reconstruction_dir,
            dense_gt_reconstruction_dir=dense_gt_reconstruction_dir,
        )
        written.append(out_file)

    return written

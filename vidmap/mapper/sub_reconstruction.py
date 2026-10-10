"""Covisibility graph analysis and decomposition of a reconstruction into sub-reconstructions at video cuts."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path

import pycolmap

logger = logging.getLogger(__name__)


def build_covisibility_graph(
    reconstruction: pycolmap.Reconstruction,
    *,
    only_registered: bool = True,
) -> tuple[dict[tuple[int, int], int], dict[int, dict[int, int]]]:
    """Build the 3D-point covisibility graph between images.

    Args:
        reconstruction: Source pycolmap reconstruction.
        only_registered: If True, only include images that have registered poses.

    Returns:
        edge_weights: dict mapping canonical pair (min_id, max_id) -> count of shared 3D points.
        adjacency: dict mapping image_id -> dict of {neighbor_id: shared_point_count}.
    """
    valid_image_ids = set()
    for image_id, image in reconstruction.images.items():
        if not only_registered or image.has_pose:
            valid_image_ids.add(int(image_id))

    edge_weights: dict[tuple[int, int], int] = defaultdict(int)
    adjacency: dict[int, dict[int, int]] = {iid: {} for iid in valid_image_ids}

    for point in reconstruction.points3D.values():
        observed_ids = [int(el.image_id) for el in point.track.elements if int(el.image_id) in valid_image_ids]
        n_obs = len(observed_ids)
        if n_obs < 2:
            continue
        for i in range(n_obs):
            u = observed_ids[i]
            for j in range(i + 1, n_obs):
                v = observed_ids[j]
                pair = (min(u, v), max(u, v))
                edge_weights[pair] += 1

    for (u, v), weight in edge_weights.items():
        adjacency[u][v] = weight
        adjacency[v][u] = weight

    return edge_weights, adjacency


def detect_covisibility_components(
    reconstruction: pycolmap.Reconstruction,
    *,
    min_shared_points: int = 20,
    min_component_size: int = 5,
) -> list[list[int]]:
    """Partition registered images into connected components based on 3D covisibility.

    Images sharing at least `min_shared_points` 3D points are connected in the
    covisibility graph. Disjoint components (e.g. separated by video shot cuts
    or tracking breaks) are identified via connected component search.

    Args:
        reconstruction: Source reconstruction.
        min_shared_points: Minimum number of shared 3D points required to retain an edge.
        min_component_size: Minimum number of images required for a component to be kept.

    Returns:
        A list of components, where each component is a sorted list of image IDs.
        Sorted chronologically by earliest image ID.
    """
    edge_weights, adjacency = build_covisibility_graph(reconstruction, only_registered=True)
    all_nodes = sorted(adjacency.keys())
    if not all_nodes:
        return []

    # Build pruned adjacency graph
    pruned_adj: dict[int, set[int]] = defaultdict(set)
    for (u, v), weight in edge_weights.items():
        if weight >= min_shared_points:
            pruned_adj[u].add(v)
            pruned_adj[v].add(u)

    visited = set()
    components = []

    for start_node in all_nodes:
        if start_node in visited:
            continue
        component = []
        queue = deque([start_node])
        visited.add(start_node)

        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in pruned_adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(component) >= min_component_size:
            components.append(sorted(component))
        else:
            logger.debug(
                "Discarding small component with %d images (< min_size %d): %s",
                len(component),
                min_component_size,
                component,
            )

    components.sort(key=lambda comp: comp[0] if comp else 0)
    return components


def extract_sub_reconstruction(
    source: pycolmap.Reconstruction,
    image_ids: Sequence[int],
    *,
    min_track_length: int = 2,
) -> pycolmap.Reconstruction:
    """Extract an isolated, valid pycolmap.Reconstruction for a subset of image IDs.

    Args:
        source: Base reconstruction.
        image_ids: Sequence of image IDs to keep.
        min_track_length: Minimum number of observations within the subset for a 3D point.

    Returns:
        A new standalone pycolmap.Reconstruction.
    """
    keep_set = {int(iid) for iid in image_ids}
    sub = pycolmap.Reconstruction()

    # 1. Cameras
    used_camera_ids = {source.images[iid].camera_id for iid in keep_set if iid in source.images}
    for cid in used_camera_ids:
        sub.add_camera(source.cameras[cid])

    # 2. Frames
    keep_frame_ids = {source.images[iid].frame_id for iid in keep_set if iid in source.images}

    # 3. Rigs
    used_rig_ids = set()
    for fid in keep_frame_ids:
        if fid in source.frames:
            frame = source.frames[fid]
            if hasattr(frame, "has_rig_id") and frame.has_rig_id():
                used_rig_ids.add(frame.rig_id)
            elif hasattr(frame, "rig_id") and frame.rig_id is not None:
                used_rig_ids.add(frame.rig_id)

    for rid, rig in source.rigs.items():
        if rid in used_rig_ids:
            continue
        try:
            sensors = (
                rig.sensor_ids() if callable(getattr(rig, "sensor_ids", None)) else getattr(rig, "sensor_ids", [])
            )
            if sensors and all(s.id in used_camera_ids for s in sensors):
                used_rig_ids.add(rid)
        except Exception:
            pass

    for rid in used_rig_ids:
        if rid in source.rigs:
            sub.add_rig(source.rigs[rid])

    for fid in keep_frame_ids:
        frame = source.frames[fid]
        frame.reset_rig_ptr()
        sub.add_frame(frame)

    # 4. Images
    for iid in keep_set:
        if iid not in source.images:
            continue
        image = source.images[iid]
        image.reset_camera_ptr()
        image.reset_frame_ptr()
        sub.add_image(image)

    # 5. Register frames
    for fid in keep_frame_ids:
        sub.register_frame(fid)

    # 6. Points3D
    for pid, pt in source.points3D.items():
        sub_elements = [el for el in pt.track.elements if int(el.image_id) in keep_set]
        if len(sub_elements) >= min_track_length:
            new_track = pycolmap.Track()
            for el in sub_elements:
                new_track.add_element(el)
            sub_pt = pycolmap.Point3D(
                xyz=pt.xyz,
                track=new_track,
                color=pt.color,
                error=pt.error,
            )
            sub.add_point3D_with_id(pid, sub_pt)

    return sub


def decompose_reconstruction(
    source: pycolmap.Reconstruction,
    *,
    min_shared_points: int = 20,
    min_model_size: int = 5,
    min_track_length: int = 2,
) -> list[pycolmap.Reconstruction]:
    """Decompose a reconstruction into multiple sub-reconstructions if cuts or weak edges exist.

    Args:
        source: Base reconstruction.
        min_shared_points: Edge pruning threshold on 3D shared points.
        min_model_size: Minimum registered images required for each sub-model.
        min_track_length: Minimum observations required to keep a 3D point in a sub-model.

    Returns:
        List of sub-reconstructions. If no cuts / only 1 component found, returns [source].
    """
    components = detect_covisibility_components(
        source,
        min_shared_points=min_shared_points,
        min_component_size=min_model_size,
    )

    if len(components) <= 1:
        logger.info(
            "Reconstruction is a single component (%d images)",
            source.num_reg_images(),
        )
        return [source]

    logger.info(
        "Decomposing reconstruction into %d sub-models (sizes: %s)",
        len(components),
        [len(c) for c in components],
    )

    sub_models = []
    for idx, component_ids in enumerate(components):
        sub_rec = extract_sub_reconstruction(
            source,
            component_ids,
            min_track_length=min_track_length,
        )
        logger.info(
            "Sub-model %d: %d images, %d 3D points",
            idx,
            sub_rec.num_reg_images(),
            sub_rec.num_points3D(),
        )
        sub_models.append(sub_rec)

    return sub_models


def export_sub_reconstructions(
    reconstructions: Sequence[pycolmap.Reconstruction],
    output_dir: str | Path,
) -> list[Path]:
    """Export one or more reconstructions to disk following COLMAP conventions.

    If len(reconstructions) == 1:
        Writes to output_dir / "rec".
    If len(reconstructions) > 1:
        Writes to output_dir / "rec" / "0", output_dir / "rec" / "1", ...
        and creates a symlink output_dir / "rec_0" -> "rec/0".

    Returns:
        List of written reconstruction directory paths.
    """
    base_dir = Path(output_dir).expanduser().resolve()
    rec_root = base_dir / "rec"

    if len(reconstructions) == 1:
        rec_root.mkdir(parents=True, exist_ok=True)
        reconstructions[0].write(rec_root)
        logger.info("Wrote single reconstruction to %s", rec_root)
        return [rec_root]

    written_paths = []
    rec_root.mkdir(parents=True, exist_ok=True)

    for idx, sub_rec in enumerate(reconstructions):
        sub_dir = rec_root / str(idx)
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_rec.write(sub_dir)
        logger.info(
            "Wrote sub-model %d (%d images, %d points) to %s",
            idx,
            sub_rec.num_reg_images(),
            sub_rec.num_points3D(),
            sub_dir,
        )
        written_paths.append(sub_dir)

    return written_paths

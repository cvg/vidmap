"""Tests for rotation cycle consistency filtering."""

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.stages.relative_pose.cycle_filter import filter_pairs_by_cycle_consistency


def _build_test_solve_state(
    rotations: list[np.ndarray], corrupt_pairs: dict[tuple[int, int], np.ndarray] | None = None
) -> SolveState:
    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(pycolmap.Camera.create_from_model_name(1, "PINHOLE", 500.0, 640, 480))
    sidecars = native.MappingSidecars()
    graph = pycolmap.PoseGraph()

    num_images = len(rotations)
    for i in range(1, num_images + 1):
        rec.add_image_with_trivial_frame(pycolmap.Image(image_id=i, camera_id=1, name=f"frame_{i:04d}.jpg"))
        sidecars.add_image(i, native.ImageData())

    for i in range(1, num_images + 1):
        for j in range(i + 1, num_images + 1):
            if corrupt_pairs and (i, j) in corrupt_pairs:
                R_rel = corrupt_pairs[(i, j)]
            else:
                R_rel = rotations[j - 1] @ rotations[i - 1].T
            pair = native.PairData()
            pair.has_relative_pose = True
            sidecars.add_pair(pycolmap.image_pair_to_pair_id(i, j), pair)
            edge = pycolmap.PoseGraphEdge()
            edge.valid = True
            edge.cam2_from_cam1 = pycolmap.Rigid3d(rotation=pycolmap.Rotation3d(R_rel))
            graph.add_edge(i, j, edge)

    return SolveState(rec, graph, sidecars)


def test_cycle_filter_consistent_graph():
    rots = [
        Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix(),
        Rotation.from_euler("xyz", [10, 0, 0], degrees=True).as_matrix(),
        Rotation.from_euler("xyz", [10, 15, 0], degrees=True).as_matrix(),
        Rotation.from_euler("xyz", [0, 15, 0], degrees=True).as_matrix(),
    ]
    state = _build_test_solve_state(rots)

    num_filtered = filter_pairs_by_cycle_consistency(
        state,
        min_triangles=2,
        max_median_cycle_error_deg=10.0,
        max_inconsistent_ratio=0.5,
        triangle_error_threshold_deg=5.0,
    )
    assert num_filtered == 0


def test_cycle_filter_inconsistent_edge():
    rots = [
        Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix(),
        Rotation.from_euler("xyz", [10, 0, 0], degrees=True).as_matrix(),
        Rotation.from_euler("xyz", [10, 15, 0], degrees=True).as_matrix(),
        Rotation.from_euler("xyz", [0, 15, 0], degrees=True).as_matrix(),
    ]
    # Corrupt edge (1, 2) with 80 degree bogus rotation
    corrupt_R = Rotation.from_euler("xyz", [80, 0, 0], degrees=True).as_matrix()
    state = _build_test_solve_state(rots, corrupt_pairs={(1, 2): corrupt_R})

    num_filtered = filter_pairs_by_cycle_consistency(
        state,
        min_triangles=2,
        max_median_cycle_error_deg=20.0,
        max_inconsistent_ratio=0.6,
        triangle_error_threshold_deg=15.0,
    )
    assert num_filtered == 1
    corrupt_pid = pycolmap.image_pair_to_pair_id(1, 2)
    assert not state.pose_graph.is_valid(corrupt_pid)

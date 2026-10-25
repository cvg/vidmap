"""Rotation averaging and evaluation operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pyceres
import pycolmap
from scipy.cluster.hierarchy import DisjointSet
from vidmap_native import global_positioning as gp_costs

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.view_graph import RAOptions
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.mapper.replay.evidence.canonical import rotation_artifact, snapshot_images
from vidmap.mapper.replay.evidence.stages import ra_summary

logger = logging.getLogger(__name__)


def add_fixed_rotation_sensors(problem, reconstruction, sensors):
    for rig_id, camera_id in sensors:
        rig = reconstruction.rig(rig_id)
        sensor_id = reconstruction.camera(camera_id).sensor_id
        if sensor_id == rig.ref_sensor_id:
            continue
        rotation = rig.non_ref_sensors[sensor_id].rotation.quat
        if not problem.has_parameter_block(rotation):
            problem.add_parameter_block(rotation, 4)
            problem.set_parameter_block_constant(rotation)


@dataclass(kw_only=True)
class RotationAverager:
    solve_state: SolveState
    options: RAOptions
    sequence_id_to_index: dict[int, int]
    filtered_consecutive_pair_ids: set[int]
    replay: ReplayCache

    def run_pass(self) -> bool:
        rec = self.solve_state.reconstruction
        gp_costs.validate_fixed_rigs(rec)
        graph, tracking_graph = self.build_graphs()
        active_frames = graph.largest_connected_frame_component(rec, self.options.filter_unregistered_images)
        if not active_frames:
            return False
        active_images = {i for i, image in rec.images.items() if image.frame_id in active_frames}
        graph.invalidate_pairs_outside_active_image_ids(active_images)
        tracking_graph.invalidate_pairs_outside_active_image_ids(active_images)
        self.bridge_tracking_gaps(graph, tracking_graph, active_frames)
        tracking_graph.add_rig_edges(rec, active_images, refine_sensor_from_rig=False)
        keep_frames = {fid for fid, frame in rec.frames.items() if frame.has_pose()}
        if not self.solve(graph, tracking_graph):
            return False
        return self.finalize(graph, keep_frames)

    def build_graphs(self) -> tuple[pycolmap.PoseGraph, pycolmap.PoseGraph]:
        state = self.solve_state
        graph = pycolmap.PoseGraph()
        tracking_graph = pycolmap.PoseGraph()
        for pair_id, edge in state.pose_graph.edges.items():
            pair = state.pair_data(pair_id)
            if not edge.valid or not pair.has_relative_pose:
                continue
            image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
            if state.image(image_id1).frame_id == state.image(image_id2).frame_id:
                continue
            inliers = pair.inlier_indices
            is_tracking = len(inliers) > 0 and 2 * np.count_nonzero(pair.are_loop_closure[inliers]) <= len(inliers)
            graph.edges[pair_id] = edge
            graph.edges[pair_id].num_matches = len(inliers)
            if is_tracking:
                tracking_graph.edges[pair_id] = graph.edges[pair_id]
        return graph, tracking_graph

    def bridge_tracking_gaps(
        self, graph: pycolmap.PoseGraph, tracking_graph: pycolmap.PoseGraph, active_frames: set[int]
    ) -> None:
        """Temporary workaround: COLMAP RA requires a connected graph before we can add loop closures."""
        rec = self.solve_state.reconstruction
        components = DisjointSet(active_frames)
        for pair_id, edge in tracking_graph.edges.items():
            if edge.valid:
                image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
                components.merge(rec.image(image_id1).frame_id, rec.image(image_id2).frame_id)
        if components.n_subsets <= 1:
            return

        def temporal_gap(pair_id):
            image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
            return abs(self.sequence_id_to_index[image_id1] - self.sequence_id_to_index[image_id2])

        for pair_id in sorted(graph.edges, key=temporal_gap):
            edge = graph.edges[pair_id]
            if not edge.valid:
                continue
            image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
            if components.merge(rec.image(image_id1).frame_id, rec.image(image_id2).frame_id):
                tracking_graph.edges[pair_id] = edge
                if components.n_subsets == 1:
                    break

    def solve(self, graph: pycolmap.PoseGraph, tracking_graph: pycolmap.PoseGraph) -> bool:
        options = pycolmap.RotationEstimatorOptions()
        options.ceres.loss_function_scale = np.rad2deg(self.options.video_tracking_huber_scale)
        options.refine_sensor_from_rig = False
        options.ceres.solver_options.max_num_iterations = self.options.max_iterations
        options.ceres.solver_options.num_threads = 1 if self.options.num_threads is None else self.options.num_threads
        loop_loss = pyceres.CauchyLoss(self.options.video_lc_cauchy_scale)
        loop_edges = {
            pair_id: edge
            for pair_id, edge in graph.edges.items()
            if edge.valid and pair_id not in tracking_graph.edges and not self.options.filter_risky_loop_closure_pairs
        }
        rec = self.solve_state.reconstruction
        loop_images = {i for pair_id in loop_edges for i in pycolmap.pair_id_to_image_pair(pair_id)}
        loop_sensors = {(rec.image(i).frame.rig_id, rec.image(i).camera_id) for i in loop_images}
        # Warm up with Huber on tracking edges, then refine with Cauchy on all edges.
        for loss_type in (pycolmap.LossFunctionType.HUBER, pycolmap.LossFunctionType.CAUCHY):
            options.ceres.loss_function_type = loss_type
            averager = pycolmap.create_default_ceres_rotation_averager(
                options, tracking_graph, self.solve_state.reconstruction
            )
            add_fixed_rotation_sensors(averager.problem, rec, loop_sensors)
            for pair_id, edge in loop_edges.items():
                image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
                averager.add_relative_rotation_residual(image_id1, image_id2, edge.cam2_from_cam1.rotation, loop_loss)
            if not averager.solve().IsSolutionUsable():
                return False
            options.skip_initialization = True
        return True

    def finalize(self, graph: pycolmap.PoseGraph, keep_frames: set[int]) -> bool:
        rec = self.solve_state.reconstruction
        if self.options.max_rotation_error_deg > 0:
            native.filter_edges_by_relative_rotation(graph, rec, self.options.max_rotation_error_deg)
            active_frames = graph.largest_connected_frame_component(rec)
            if not active_frames:
                return False
            keep_frames.intersection_update(active_frames)
        for frame_id in rec.reg_frame_ids():
            if frame_id not in keep_frames:
                rec.deregister_frame(frame_id)
        return True

    def average(self) -> None:
        state = self.solve_state
        rec = self.solve_state.reconstruction
        for _ in range(2 if self.options.max_rotation_error_deg > 0.0 else 1):
            if not self.run_pass():
                raise RuntimeError("Rotation averaging failed")
        logger.info(f"{len(rec.reg_image_ids())} are within the connected component.")
        if self.replay.write_enabled("ra"):
            images = snapshot_images(state)
            summary = ra_summary(state, images, self.filtered_consecutive_pair_ids)
            self.replay.write_json("ra", "rotations.json", rotation_artifact(images))
            self.replay.write_json("ra", "summary.json", summary)

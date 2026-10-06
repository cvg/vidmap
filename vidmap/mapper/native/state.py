"""Canonical native mapper state and explicit pycolmap checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pycolmap

from .extension import native
from .records import camera_record_from_pycolmap, pose_record_from_pycolmap, pose_record_to_pycolmap


class SolveState:
    """Own the canonical native problem and its stock pycolmap checkpoint."""

    def __init__(
        self,
        reconstruction: pycolmap.Reconstruction,
        native_problem: native.MappingProblem,
        *,
        image_order: Sequence[int] | None = None,
        pair_order: Sequence[int] | None = None,
    ) -> None:
        self.reconstruction = reconstruction
        self.native_problem = native_problem
        source_image_order = native_problem.image_ids if image_order is None else image_order
        source_pair_order = native_problem.pair_ids if pair_order is None else pair_order
        self.image_order = [int(value) for value in source_image_order]
        self.pair_order = [int(value) for value in source_pair_order]
        self._validate_order(self.image_order, native_problem.image_ids, "image_order")
        self._validate_order(self.pair_order, native_problem.pair_ids, "pair_order")
        self.native_problem.validate()

        self.imu_edges: list[native.ImuEdgeRecord] = []
        self.imu_states: dict[int, native.ImuStateRecord] = {}
        self.imu_from_cam: native.PoseRecord = native.PoseRecord()
        self.initial_imu_from_cam: native.PoseRecord = native.PoseRecord()
        self.gravity_direction: np.ndarray | None = None
        self.gravity_in_world: np.ndarray | None = None
        self.imu_scale: float | None = None
        self.has_metric_imu_scale: bool = False

    def set_imu_data(
        self,
        imu_edges: Sequence[native.ImuEdgeRecord],
        *,
        imu_from_cam: native.PoseRecord | pycolmap.Rigid3d | None = None,
        imu_states: Sequence[native.ImuStateRecord] | Mapping[int, native.ImuStateRecord] | None = None,
    ) -> None:
        self.imu_edges = list(imu_edges)
        if imu_from_cam is not None:
            if isinstance(imu_from_cam, pycolmap.Rigid3d):
                pose_rec = pose_record_from_pycolmap(imu_from_cam)
            else:
                pose_rec = native.PoseRecord()
                pose_rec.has_pose = bool(imu_from_cam.has_pose)
                pose_rec.rotation_xyzw = np.asarray(imu_from_cam.rotation_xyzw, dtype=np.float64).copy()
                pose_rec.translation = np.asarray(imu_from_cam.translation, dtype=np.float64).copy()
            self.imu_from_cam = pose_rec
            init_rec = native.PoseRecord()
            init_rec.has_pose = bool(pose_rec.has_pose)
            init_rec.rotation_xyzw = np.asarray(pose_rec.rotation_xyzw, dtype=np.float64).copy()
            init_rec.translation = np.asarray(pose_rec.translation, dtype=np.float64).copy()
            self.initial_imu_from_cam = init_rec
        if imu_states is not None:
            self.update_imu_states(imu_states)

    def imu_states_list(self) -> list[native.ImuStateRecord]:
        return [self.imu_states[image_id] for image_id in self.image_order if image_id in self.imu_states]

    def update_imu_states(
        self,
        states: Mapping[int, native.ImuStateRecord] | Sequence[native.ImuStateRecord],
        *,
        use_metric_velocity: bool = False,
    ) -> None:
        items = states.values() if isinstance(states, Mapping) else states
        updated: dict[int, native.ImuStateRecord] = {}
        for record in items:
            cloned = native.ImuStateRecord()
            cloned.image_id = int(record.image_id)
            vel = record.metric_velocity if use_metric_velocity else record.velocity
            cloned.velocity = np.asarray(vel, dtype=np.float64).copy()
            cloned.metric_velocity = np.asarray(record.metric_velocity, dtype=np.float64).copy()
            cloned.bias_gyro = np.asarray(record.bias_gyro, dtype=np.float64).copy()
            cloned.bias_accel = np.asarray(record.bias_accel, dtype=np.float64).copy()
            updated[int(record.image_id)] = cloned
        self.imu_states = updated

    def reintegrate_imu_edges(self) -> None:
        if not self.imu_edges or not self.imu_states:
            return
        for edge in self.imu_edges:
            if not edge.has_integrator:
                continue
            state = self.imu_states.get(int(edge.image_id1))
            if state is None:
                continue
            biases = np.concatenate([state.bias_gyro, state.bias_accel]).astype(np.float64)
            if np.linalg.norm(biases - np.asarray(edge.data.biases, dtype=np.float64)) > 1e-6:
                edge.reintegrate(biases)

    @staticmethod
    def _validate_order(order: Sequence[int], identifiers: Sequence[int], label: str) -> None:
        if len(order) != len(set(order)):
            raise ValueError(f"{label} contains duplicates")
        if set(order) != set(identifiers):
            raise ValueError(f"{label} must contain every native record exactly once")

    def image(self, image_id: int) -> native.ImageRecord:
        return self.native_problem.image(int(image_id))

    def update_image(self, record: native.ImageRecord) -> None:
        self.native_problem.update_image(record)

    def image_records(self) -> dict[int, native.ImageRecord]:
        return {image_id: self.image(image_id) for image_id in self.image_order}

    def pair(self, pair_id: int) -> native.PairRecord:
        return self.native_problem.pair(int(pair_id))

    def update_pair(self, record: native.PairRecord) -> None:
        self.native_problem.update_pair(record)

    def export_pair_pose(self, pair_id: int) -> pycolmap.Rigid3d | None:
        return pose_record_to_pycolmap(self.pair(pair_id).geometry.cam2_from_cam1)

    def pair_records(self) -> dict[int, native.PairRecord]:
        return {pair_id: self.pair(pair_id) for pair_id in self.pair_order}

    def track_records(self) -> dict[int, native.TrackRecord]:
        return {
            int(point3D_id): self.native_problem.track(point3D_id) for point3D_id in self.native_problem.point3D_ids
        }

    def replace_tracks(self, tracks: Sequence[native.TrackRecord]) -> None:
        self.native_problem.clear_tracks()
        for track in tracks:
            self.native_problem.add_track(track)

    def import_cameras(self) -> None:
        for camera_id in self.native_problem.camera_ids:
            if camera_id in self.reconstruction.cameras:
                self.native_problem.update_camera(camera_record_from_pycolmap(self.reconstruction.camera(camera_id)))

    def export_cameras(self) -> None:
        for camera_id in self.native_problem.camera_ids:
            if camera_id not in self.reconstruction.cameras:
                continue
            record = self.native_problem.camera(camera_id)
            camera = self.reconstruction.camera(camera_id)
            if int(camera.model) != int(record.model_id):
                raise ValueError(f"camera {camera_id} model changed across the native boundary")
            camera.params = np.asarray(record.params, dtype=np.float64)
            camera.has_prior_focal_length = bool(record.has_prior_focal_length)

    def export_image_pose(self, image_id: int) -> None:
        image_id = int(image_id)
        if image_id not in self.reconstruction.images:
            return
        record = self.image(image_id)
        frame_id = int(self.reconstruction.image(image_id).frame_id)
        registered_frames = set(self.reconstruction.reg_frame_ids())
        if record.pose.has_pose:
            pose = pose_record_to_pycolmap(record.pose)
            self.reconstruction.frame(frame_id).set_cam_from_world(record.camera_id, pose)
            if frame_id not in registered_frames:
                self.reconstruction.register_frame(frame_id)
        elif frame_id in registered_frames:
            self.reconstruction.deregister_frame(frame_id)

    def export_poses(self) -> None:
        for image_id in self.image_order:
            self.export_image_pose(image_id)

    def import_poses(self) -> None:
        for image_id in self.image_order:
            record = self.image(image_id)
            if image_id in self.reconstruction.images:
                image = self.reconstruction.image(image_id)
                record.pose = pose_record_from_pycolmap(image.cam_from_world() if image.has_pose else None)
            else:
                record.pose = native.PoseRecord()
            self.update_image(record)

    def export_tracks(self) -> None:
        for point3D_id in list(self.reconstruction.point3D_ids()):
            self.reconstruction.delete_point3D(point3D_id)
        for point3D_id, record in self.track_records().items():
            elements = [
                pycolmap.TrackElement(int(image_id), int(point2D_idx))
                for image_id, point2D_idx in np.asarray(record.observations)
                if int(image_id) in self.reconstruction.images
            ]
            point = pycolmap.Point3D(
                xyz=np.asarray(record.xyz, dtype=np.float64),
                color=np.asarray(record.color, dtype=np.uint8),
                error=float(record.error),
                track=pycolmap.Track(elements),
            )
            self.reconstruction.add_point3D_with_id(point3D_id, point)

    def export_track_values(self) -> None:
        records = self.track_records()
        if set(records) != set(int(value) for value in self.reconstruction.point3D_ids()):
            self.export_tracks()
            return
        for point3D_id, record in records.items():
            point = self.reconstruction.point3D(point3D_id)
            point.xyz = np.asarray(record.xyz, dtype=np.float64)
            point.color = np.asarray(record.color, dtype=np.uint8)
            point.error = float(record.error)

    def import_tracks(self) -> None:
        loop_closure_sidecars = {
            int(point3D_id): (
                np.asarray(record.loop_closure_observations, dtype=np.uint32).copy(),
                np.asarray(record.loop_closure_anchors, dtype=np.uint32).copy(),
            )
            for point3D_id, record in self.track_records().items()
        }
        self.native_problem.clear_tracks()
        for point3D_id, point in self.reconstruction.points3D.items():
            record = native.TrackRecord(int(point3D_id), point)
            point3D_id = int(point3D_id)
            if point3D_id in loop_closure_sidecars:
                loop_closure_observations, loop_closure_anchors = loop_closure_sidecars[point3D_id]
            else:
                loop_closure_observations = np.empty((0, 2), dtype=np.uint32)
                loop_closure_anchors = np.empty((0, 2), dtype=np.uint32)
            record.loop_closure_observations = loop_closure_observations
            record.loop_closure_anchors = loop_closure_anchors
            self.native_problem.add_track(record)

    def export_scene(self, *, tracks: bool = True) -> None:
        self.export_cameras()
        self.export_poses()
        if tracks:
            self.export_tracks()

    def import_scene(self, *, tracks: bool = True) -> None:
        self.import_cameras()
        self.import_poses()
        if tracks:
            self.import_tracks()

    def import_checkpoint(self, reconstruction: pycolmap.Reconstruction) -> None:
        prior_focal_flags = {
            int(camera_id): bool(self.native_problem.camera(camera_id).has_prior_focal_length)
            for camera_id in self.native_problem.camera_ids
        }
        known_image_ids = set(self.native_problem.image_ids)
        for image_id, image in reconstruction.images.items():
            if image_id not in known_image_ids:
                raise ValueError(f"checkpoint contains unknown image {image_id}")
            record = self.image(image_id)
            if image.name != record.name:
                raise ValueError(f"checkpoint image {image_id} has a different name")
            if image.num_points2D() != len(record.keypoints):
                raise ValueError(f"checkpoint image {image_id} has a different feature count")
        self.reconstruction = reconstruction
        for camera_id, has_prior_focal_length in prior_focal_flags.items():
            if camera_id in reconstruction.cameras:
                reconstruction.camera(camera_id).has_prior_focal_length = has_prior_focal_length
        self.import_scene()

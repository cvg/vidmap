"""COLMAP scene ownership and VidMap observation metadata."""

from __future__ import annotations

from collections.abc import Sequence

import pycolmap

from .extension import native


class SolveState:
    def __init__(
        self,
        reconstruction: pycolmap.Reconstruction,
        pose_graph: pycolmap.PoseGraph,
        sidecars: native.MappingSidecars,
        *,
        image_order: Sequence[int] | None = None,
        pair_order: Sequence[int] | None = None,
    ) -> None:
        self.reconstruction = reconstruction
        self.pose_graph = pose_graph
        self.sidecars = sidecars
        self.image_order = list(reconstruction.images if image_order is None else image_order)
        self.pair_order = list(pose_graph.edges if pair_order is None else pair_order)
        self._validate_order(self.image_order, reconstruction.images, "image_order")
        self._validate_order(self.pair_order, pose_graph.edges, "pair_order")
        self.retriangulation_graph = None
        self.replay_graph_summary = None
        self.replay_pose_graph_summary = None

    @staticmethod
    def _validate_order(order, identifiers, label):
        if len(order) != len(set(order)) or set(order) != set(identifiers):
            raise ValueError(f"{label} must contain every identifier exactly once")

    def image(self, image_id: int) -> pycolmap.Image:
        return self.reconstruction.image(image_id)

    def image_data(self, image_id: int):
        return self.sidecars.image(image_id)

    def pair_data(self, pair_id: int):
        return self.sidecars.pair(pair_id)

    def import_checkpoint(self, reconstruction: pycolmap.Reconstruction) -> None:
        for image_id, image in reconstruction.images.items():
            if image_id not in self.reconstruction.images:
                raise ValueError(f"checkpoint contains unknown image {image_id}")
            original = self.image(image_id)
            if image.name != original.name:
                raise ValueError(f"checkpoint image {image_id} has a different name")
            if image.num_points2D() != original.num_points2D():
                raise ValueError(f"checkpoint image {image_id} has a different feature count")
        for camera_id, camera in self.reconstruction.cameras.items():
            if camera_id in reconstruction.cameras:
                reconstruction.camera(camera_id).has_prior_focal_length = camera.has_prior_focal_length
        if self.retriangulation_graph is not None and reconstruction.num_images() < self.reconstruction.num_images():
            self.retriangulation_graph = native.filter_correspondence_graph(
                self.retriangulation_graph, reconstruction, self.pair_order
            )
        self.reconstruction = reconstruction

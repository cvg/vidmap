"""Render saved geometry as interactive packed Three.js HTML."""

from __future__ import annotations

import copy
import logging
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vidmap.datasets.base import image_has_public_pose
from vidmap.utils.trajectory import align_reconstruction_to_reference_sequence

from . import threejs_scene
from .point_quality import lowest_covariance_point_ids

logger = logging.getLogger(__name__)

_PAIR_ID_BASE = 2147483647
_VALID_TWO_VIEW_CONFIGS = frozenset({2, 3, 4, 5, 6, 9})


def _database_loop_closure_names(database: Path) -> tuple[tuple[str, str], ...]:
    uri = f"{database.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        images = {
            int(image_id): str(name) for image_id, name in connection.execute("SELECT image_id, name FROM images")
        }
        pair_rows = tuple(
            connection.execute("SELECT pair_id, config FROM two_view_geometries WHERE rows > 0 ORDER BY pair_id")
        )
    pairs = []
    for pair_id, config in pair_rows:
        if config is None or int(config) not in _VALID_TWO_VIEW_CONFIGS:
            continue
        pair_id = int(pair_id)
        first_id = (pair_id - pair_id % _PAIR_ID_BASE) // _PAIR_ID_BASE
        second_id = pair_id % _PAIR_ID_BASE
        if first_id not in images or second_id not in images or first_id == second_id:
            continue
        pairs.append((images[first_id], images[second_id]))
    return tuple(pairs)


def _reconstructed_point_ids(image) -> np.ndarray:
    values = np.fromiter(
        (int(point.point3D_id) for point in image.points2D if point.has_point3D()),
        dtype=np.uint64,
    )
    return np.unique(values)


def _ground_truth_for_model(ground_truth, model):
    """Retain exact, effective ground-truth poses present in the estimate."""
    estimated_names = {image.name for image in model.images.values() if image.has_pose}
    posed_ground_truth = [image for image in ground_truth.images.values() if image.has_pose]
    if not posed_ground_truth:
        return ground_truth
    matched = [image for image in posed_ground_truth if image.name in estimated_names]
    if not matched:
        raise ValueError("Ground truth has no posed image names in common with the estimated reconstruction")
    selected = [image for image in matched if image_has_public_pose(image)]

    import pycolmap

    subset = pycolmap.Reconstruction()
    camera_ids = sorted({image.camera_id for image in selected})
    for camera_id in camera_ids:
        subset.add_camera_with_trivial_rig(ground_truth.cameras[camera_id])
    for source in sorted(selected, key=lambda image: image.image_id):
        image = pycolmap.Image(
            image_id=source.image_id,
            camera_id=source.camera_id,
            name=source.name,
        )
        subset.add_image_with_trivial_frame(image, source.cam_from_world())
    return subset


@dataclass(frozen=True)
class InteractiveHtmlExporter:
    """Export one reconstruction and optional ground truth to interactive HTML."""

    model: object
    ground_truth: object | None = None
    database: Path | None = None
    include_cameras: bool = False
    point_covariance_percentile: float | None = None

    def __post_init__(self) -> None:
        if self.point_covariance_percentile is not None and not (0 < self.point_covariance_percentile <= 100):
            raise ValueError("point covariance percentile must be in (0, 100]")

    @classmethod
    def from_paths(
        cls,
        reconstruction: str | Path,
        *,
        ground_truth: str | Path | None = None,
        database: str | Path | None = None,
        include_cameras: bool = False,
        point_covariance_percentile: float | None = None,
    ) -> InteractiveHtmlExporter:
        """Load COLMAP models from disk and create an exporter."""
        import pycolmap

        reconstruction_path = Path(reconstruction).expanduser()
        if not reconstruction_path.is_dir():
            raise FileNotFoundError(f"Reconstruction directory does not exist: {reconstruction_path}")
        model = pycolmap.Reconstruction(reconstruction_path)

        ground_truth_model = None
        if ground_truth is not None:
            ground_truth_path = Path(ground_truth).expanduser()
            if not ground_truth_path.is_dir():
                raise FileNotFoundError(f"Ground-truth directory does not exist: {ground_truth_path}")
            ground_truth_model = pycolmap.Reconstruction(ground_truth_path)

        resolved_database = None
        if database is not None:
            resolved_database = Path(database).expanduser()
            if not resolved_database.is_file():
                raise FileNotFoundError(f"COLMAP database does not exist: {resolved_database}")
        return cls(
            model=model,
            ground_truth=ground_truth_model,
            database=resolved_database,
            include_cameras=include_cameras,
            point_covariance_percentile=point_covariance_percentile,
        )

    def build_scene(self) -> threejs_scene.SceneGeometry:
        """Build packed scene geometry without mutating caller-owned models."""
        display_model = copy.deepcopy(self.model)
        display_ground_truth = copy.deepcopy(self.ground_truth)
        if display_ground_truth is not None and display_ground_truth.num_reg_images() > 0:
            display_ground_truth = _ground_truth_for_model(display_ground_truth, display_model)
            # Match the benchmark's frozen two-pass alignment so HTML written
            # during mapping and HTML regenerated from the saved model share
            # the same ground-truth coordinate system.
            if display_ground_truth.num_reg_images() > 0:
                for _ in range(2):
                    _, transform = align_reconstruction_to_reference_sequence(
                        display_model,
                        display_ground_truth,
                    )
                    display_model.transform(transform)
        point_ids = None
        point_colors = threejs_scene.stored_point_colors(display_model)
        if self.point_covariance_percentile is not None:
            all_point_ids = tuple(display_model.points3D)
            point_ids = lowest_covariance_point_ids(
                display_model,
                all_point_ids,
                self.point_covariance_percentile,
            )
            index_by_id = {point_id: index for index, point_id in enumerate(all_point_ids)}
            point_colors = point_colors[[index_by_id[int(point_id)] for point_id in point_ids]]
        loop_closure_edges, loop_closure_shared_points = self._loop_closure_geometry(display_model)
        return threejs_scene.build_scene_geometry(
            display_model,
            display_ground_truth,
            point_ids=point_ids,
            point_colors=point_colors,
            loop_closure_edges=loop_closure_edges,
            loop_closure_shared_points=loop_closure_shared_points,
            include_cameras=self.include_cameras,
        )

    def _loop_closure_geometry(self, model) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.database is None:
            return None, None
        model_by_name = {str(image.name): image for image in model.images.values() if image.has_pose}
        point_ids_by_name = {name: _reconstructed_point_ids(image) for name, image in model_by_name.items()}
        edges, shared_points = [], []
        for left_name, right_name in _database_loop_closure_names(self.database):
            if left_name not in model_by_name or right_name not in model_by_name:
                continue
            edges.extend(
                (
                    np.asarray(model_by_name[left_name].projection_center(), dtype=np.float32),
                    np.asarray(model_by_name[right_name].projection_center(), dtype=np.float32),
                )
            )
            shared_points.append(
                len(
                    np.intersect1d(
                        point_ids_by_name[left_name],
                        point_ids_by_name[right_name],
                        assume_unique=True,
                    )
                )
            )
        if not edges:
            return None, None
        return np.asarray(edges, dtype=np.float32), np.asarray(shared_points, dtype=np.float32)

    def render_html(self) -> str:
        """Render the complete scene HTML document."""
        return threejs_scene.render_scene_html(self.build_scene())

    def write(self, output: str | Path) -> Path:
        """Atomically write the rendered scene and return its path."""
        output = Path(output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(self.render_html())
            temporary.replace(output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        logger.info(f"Visualization saved to {output}")
        return output


def write_model_html(
    model,
    *,
    scene_parser=None,
    database: str | Path | None = None,
    include_cameras: bool = False,
    point_covariance_percentile: float | None = None,
    output: str | Path,
) -> Path:
    """Write the reconstruction pipeline's interactive HTML output."""
    ground_truth = None if scene_parser is None else scene_parser.rec
    return InteractiveHtmlExporter(
        model,
        ground_truth,
        database=None if database is None else Path(database),
        include_cameras=include_cameras,
        point_covariance_percentile=point_covariance_percentile,
    ).write(output)

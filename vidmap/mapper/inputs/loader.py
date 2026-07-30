"""Own finalized mapper-input loading and solve-state initialization."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from vidmap.mapper.inputs import MapperInputs
from vidmap.mapper.inputs.database import copy_finalized_database, load_finalized_database
from vidmap.mapper.native.extension import native
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.mapper import SetupOptions
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.mapper.replay.evidence.stages import database_file_summary, database_to_native_summary
from vidmap.utils.image_sampling import sample_at_keypoints
from vidmap.utils.io import ordered_pair_images
from vidmap.utils.loop_closure_masks import read_loop_closure_masks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MappingStageInputs:
    """Inputs consumed by the mapping-stage sequence."""

    solve_state: SolveState
    consecutive_pair_ids: list[int]
    sequence_id_to_index: dict[int, int]
    vgc_exclusion_ids: set[int]
    focal_uncertainty: float | None


def _index_lc_masks(
    lc_masks: Mapping[tuple[str, str], np.ndarray],
) -> dict[frozenset[str], np.ndarray]:
    """Index masks by undirected image pair without changing match-row order."""
    indexed = {}
    for pair, mask in lc_masks.items():
        key = frozenset(pair)
        if len(pair) != 2 or len(key) != 2:
            raise ValueError(f"Invalid LC-mask image pair: {pair!r}")
        if key in indexed:
            raise ValueError(f"Duplicate undirected LC-mask image pair: {pair!r}")
        indexed[key] = mask
    return indexed


@dataclass(kw_only=True)
class MappingProblemLoader:
    """Load one mapping problem in the established byte-reproducible order."""

    options: SetupOptions
    use_geocalib: bool
    inputs: MapperInputs
    sfm_outputs_dir: Path
    replay: ReplayCache

    def load_depth_inputs(self, depth_path: Path, images):
        """Load persisted depth priors at the reconstruction's exact features."""
        outputs = {}
        with h5py.File(str(depth_path), "r") as hfile:
            for image in images:
                image_name = image.name
                group = hfile[image_name]
                depth = group["depth"][:]
                valid = group["valid"][:]
                if depth.shape != valid.shape:
                    raise ValueError(f"{depth_path}: {image_name!r} depth and validity shapes differ")

                keypoints = np.asarray(image.keypoints)
                if valid.ndim == 2:
                    if "original_width" not in group.attrs or "original_height" not in group.attrs:
                        raise ValueError(f"{depth_path}: {image_name!r} original image size is unavailable")
                    original_width = group.attrs["original_width"]
                    original_height = group.attrs["original_height"]
                    if original_width <= 0 or original_height <= 0:
                        raise ValueError(f"{depth_path}: {image_name!r} original image size is invalid")
                    depth_h, depth_w = depth.shape[:2]
                    sx = np.float32(depth_w / original_width)
                    sy = np.float32(depth_h / original_height)
                    if len(keypoints) > 0:
                        sampled_depth = sample_at_keypoints(keypoints, depth, sx, sy)
                        sampled_valid = sample_at_keypoints(
                            keypoints,
                            valid.astype(np.float32),
                            sx,
                            sy,
                            mode="nearest",
                        ).astype(bool)
                    else:
                        sampled_depth = np.array([], dtype=np.float32)
                        sampled_valid = np.array([], dtype=bool)
                    entry = {"depth": sampled_depth, "valid": sampled_valid}
                else:
                    if depth.shape[0] != len(keypoints):
                        raise ValueError(
                            f"Depth sample count for {image_name!r} does not match reconstruction features"
                        )
                    entry = {"depth": depth, "valid": valid}
                outputs[image_name] = entry
        return outputs

    def attach_depth_inputs(self, state: SolveState, depths) -> None:
        """Attach sampled depth inputs and configured uncertainty to solve images."""
        for image_id in state.image_order:
            image = state.image(image_id)
            depth = depths[image.name]
            image.depth_values = np.asarray(depth["depth"], dtype=np.float64)
            image.depth_stddevs = np.asarray(
                depth["depth"] * self.options.depth_uncertainty_scale,
                dtype=np.float64,
            )
            image.depth_validity = np.asarray(depth["valid"], dtype=np.uint8)
            state.update_image(image)

    def load(self, *, on_inputs_validated: Callable[[], None] | None = None) -> MappingStageInputs:
        focal_uncertainty = None

        self.inputs.validate(use_geocalib=self.use_geocalib)
        if on_inputs_validated is not None:
            on_inputs_validated()
        database_path = self.inputs.database_path
        consecutive_pairs = self.inputs.read_track_pairs()
        lc_masks = read_loop_closure_masks(self.inputs.lc_masks_path)
        vgc_filtered_pairs = self.inputs.read_vgc_filtered_pairs()
        if self.use_geocalib:
            if self.inputs.geocalib_batch_path is None:
                raise FileNotFoundError("GeoCalib mapper input is required when use_geocalib=true")
            with h5py.File(self.inputs.geocalib_batch_path, "r") as hfile:
                focal_uncertainty = float(min(hfile["batch_calibration/focal_uncertainty"][:]))

        if not database_path.exists():
            raise FileNotFoundError(f"Finalized mapper database not found: {database_path}")

        # When enabled, replay captures the protected input database before
        # any native state is loaded or modified.
        if self.replay.write_enabled("database"):
            replay_database = self.replay.root / "database" / "database_complete.db"
            copy_finalized_database(database_path, replay_database)
            self.replay.write_json("database", "summary.json", database_file_summary(replay_database))

        # Mapping always mutates an output-local database copy; the mapper
        # input boundary remains read-only and reusable.
        working_database_path = Path(self.sfm_outputs_dir) / "database_complete.db"
        if database_path.resolve() == working_database_path.resolve():
            raise ValueError("Mapper inputs and outputs must use separate database paths")
        copy_finalized_database(database_path, working_database_path)
        state = load_finalized_database(working_database_path)
        if self.replay.write_enabled("db_to_glomap"):
            summary = database_to_native_summary(
                working_database_path,
                state,
                state.image_records(),
            )
            self.replay.write_json("db_to_glomap", "summary.json", summary)
        depths = self.load_depth_inputs(self.inputs.depth_maps_path, state.image_records().values())

        self.attach_depth_inputs(state, depths)

        if not state.pair_order:
            raise RuntimeError("Can't continue without image pairs")

        # Align persisted loop-closure masks with native pair records while
        # building the name-to-ID index used by later sequence projections.
        pair_name_to_pid = {}
        lc_masks_by_pair = _index_lc_masks(lc_masks)
        for pid, pair in state.pair_records().items():
            name1, name2 = (
                state.image(pair.image_id1).name,
                state.image(pair.image_id2).name,
            )
            pair_key = frozenset((name1, name2))
            pair_name_to_pid[pair_key] = pid
            if pair_key not in lc_masks_by_pair:
                raise KeyError(f"Missing LC mask for {name1} <-> {name2}")
            mask = lc_masks_by_pair[pair_key]
            if len(mask) != len(pair.all_matches):
                message = (
                    f"LC mask length mismatch for {name1} <-> {name2}: "
                    f"mask={len(mask)} matches={len(pair.all_matches)}"
                )
                raise AssertionError(message)
            pair.are_loop_closure = np.asarray(mask, dtype=np.uint8)
            state.update_pair(pair)

        # VGC exclusions are temporary pair IDs, derived only after the full
        # native pair index has been established.
        vgc_exclusion_ids = set()
        if vgc_filtered_pairs:
            for name1, name2 in vgc_filtered_pairs:
                pair_id = pair_name_to_pid.get(frozenset((name1, name2)))
                if pair_id is not None:
                    vgc_exclusion_ids.add(pair_id)
            logger.info(
                "Two-stage VGC will temporarily exclude %d pairs",
                len(vgc_exclusion_ids),
            )

        # The temporal track-pair file must describe exactly one adjacent chain
        # over the finalized database images.
        images = state.image_records()
        sequence_names = ordered_pair_images(consecutive_pairs)
        if consecutive_pairs != list(zip(sequence_names, sequence_names[1:])):
            raise ValueError("Mapper track pairs must form one ordered adjacent image chain")
        name_to_id = {image.name: image_id for image_id, image in images.items()}
        if set(sequence_names) != set(name_to_id):
            raise ValueError("Mapper track-pair sequence does not match finalized database images")
        sequence_ids = [name_to_id[name] for name in sequence_names]
        sequence_id_to_index = {sid: idx for idx, sid in enumerate(sequence_ids)}

        consecutive_pair_ids = []
        for pair in consecutive_pairs:
            pair_id = pair_name_to_pid.get(frozenset(pair))
            if pair_id is not None:
                consecutive_pair_ids.append(pair_id)

        # Finalize native pair classifications and relative-pose storage only
        # after masks, exclusions, and temporal ordering have been validated.
        native.update_image_pair_configurations(state.native_problem)
        native.decompose_relative_poses(state.native_problem)

        return MappingStageInputs(
            solve_state=state,
            consecutive_pair_ids=consecutive_pair_ids,
            sequence_id_to_index=sequence_id_to_index,
            vgc_exclusion_ids=vgc_exclusion_ids,
            focal_uncertainty=focal_uncertainty,
        )

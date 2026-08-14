"""
Streaming track frontend with composed dense matching and sparse propagation.

This module implements a streaming variant of track propagation that:
1. Computes dense matches on-the-fly as needed
2. Runs track propagation for consecutive pairs
3. Reuses the same dense fields for sequential LC matches
4. Outputs sparse features, propagated matches, and sequential LC matches

The streaming owner manages RoMa, image prefetch, decoded-image reuse, H5 I/O, and dense-field reuse.
``SparseTrackState`` owns only the numerical propagation state.
"""

import sys
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from vidmap.frontend.h5_write_queue import H5WriteQueue, save_keypoints, write_pair_matches
from vidmap.frontend.image_dataset import ImageDatasetOptions, get_image_size
from vidmap.frontend.keyframes.selection import score_keyframe_lookahead
from vidmap.frontend.paths import FrontendPaths
from vidmap.frontend.tracking.kernels import select_lc_matches_from_dense
from vidmap.frontend.tracking.multiflow import build_multiflow_window
from vidmap.frontend.tracking.state import SparseTrackState
from vidmap.frontend.video_images import RomaVideoImageDataset, load_roma_resolution_pair
from vidmap.utils.io import H5KeypointReader, get_keypoints
from vidmap.utils.keypoint_scaling import scale_keypoints
from vidmap.utils.logging import progress_bars_enabled
from vidmap.utils.parsers import names_to_pair


@dataclass(frozen=True)
class DenseMatchField:
    matches: torch.Tensor
    certainty: torch.Tensor
    covariance: torch.Tensor
    source_size: tuple[int, int]
    target_size: tuple[int, int]


@dataclass
class DenseHopBuffers:
    matches: np.ndarray
    certainty: np.ndarray
    covariance: np.ndarray

    @classmethod
    def create(cls, max_hop, current_size):
        width, height = current_size
        return cls(
            matches=np.full((max_hop, height, width, 2), -1, dtype=np.float32),
            certainty=np.full((max_hop, height, width), -1, dtype=np.float32),
            covariance=np.full((max_hop, height, width, 2, 2), -1, dtype=np.float32),
        )


class CandidateImageDataset(torch.utils.data.Dataset):
    """Load each candidate once for the dynamic propagation loop."""

    def __init__(self, highres_dataset, lowres_dataset):
        self.highres_dataset = highres_dataset
        self.lowres_dataset = lowres_dataset

    def __len__(self):
        return len(self.highres_dataset)

    def __getitem__(self, position):
        highres, lowres = load_roma_resolution_pair(self.highres_dataset, self.lowres_dataset, position)
        return highres["image"], lowres["image"]


def _write_sequential_match(
    name0,
    name1,
    dense,
    *,
    scene_parser,
    sparse_features_path,
    match_threshold,
    writer_queue,
):
    """Convert one reused dense field into a queued sequential sparse match."""
    pair_name = names_to_pair(name0, name1)
    kpts0 = get_keypoints(sparse_features_path, name0)
    kpts1 = get_keypoints(sparse_features_path, name1)
    width0, height0 = get_image_size(scene_parser, name0)
    width1, height1 = get_image_size(scene_parser, name1)
    lc_matches = select_lc_matches_from_dense(
        kpts0,
        kpts1,
        dense.matches,
        dense.certainty,
        source_size=(width0, height0),
        target_size=(width1, height1),
        lc_match_thresh=match_threshold,
    )
    writer_queue.put(
        (
            pair_name,
            {
                "matches0": torch.from_numpy(lc_matches["matches0"])[None],
                "matching_scores0": torch.from_numpy(lc_matches["matching_scores0"]).float()[None],
            },
        )
    )


def flush_sequential_matches(
    entries,
    *,
    scene_parser,
    sparse_features_path,
    match_threshold,
    writer_queue,
):
    for (name0, name1), dense in entries:
        _write_sequential_match(
            name0,
            name1,
            dense,
            scene_parser=scene_parser,
            sparse_features_path=sparse_features_path,
            match_threshold=match_threshold,
            writer_queue=writer_queue,
        )


class StreamingTrackPropagator:
    """Own streaming dense matching and apply it to composed sparse state."""

    def __init__(
        self,
        conf,
        scene_parser,
        paths: FrontendPaths,
        keyframe_sequence,
        lowres_match_resolution,
        tracker_model=None,
        conf_highres=None,
        extended_matches_path=None,
        lc_match_thresh=0.5,
        candidate_indices=None,
        forced_candidate_indices=(),
        calibrations=None,
        keyframe_options=None,
    ):
        """Store dependencies; resource acquisition starts in :meth:`run`."""
        self.conf = conf
        self.tracker_model = tracker_model
        self.conf_highres = conf_highres
        self.lowres_match_resolution = int(lowres_match_resolution)
        self.extended_matches_path = extended_matches_path
        self.lc_match_thresh = lc_match_thresh
        self.scene_parser = scene_parser
        self.paths = paths
        self.keyframe_sequence = tuple(keyframe_sequence)
        if len(self.keyframe_sequence) < 2:
            raise ValueError("Sparse propagation requires at least two frames")
        self.candidate_indices = (
            tuple(range(len(self.keyframe_sequence)))
            if candidate_indices is None
            else tuple(int(index) for index in candidate_indices)
        )
        if len(self.candidate_indices) != len(self.keyframe_sequence):
            raise ValueError("Candidate indices and names must have equal length")
        self.forced_candidate_indices = frozenset(int(index) for index in forced_candidate_indices)
        self.calibrations = {} if calibrations is None else dict(calibrations)
        self.keyframe_options = keyframe_options
        self._trackprop_first_match = True
        self._current_window_images = {}
        self._salient_keypoint_reader = None
        self._salient_keypoint_reader_context = None

    def _initialize_run(self):
        self._create_image_datasets()

    def get_keypoints(self, image_name):
        if self._salient_keypoint_reader is not None:
            return self._salient_keypoint_reader.get(image_name)
        return get_keypoints(self.paths.salient_features_path, image_name)

    def _open_salient_keypoint_reader(self):
        context = H5KeypointReader(self.paths.salient_features_path, max_size=512)
        self._salient_keypoint_reader_context = context
        self._salient_keypoint_reader = context.__enter__()

    def _close_salient_keypoint_reader(self):
        context = self._salient_keypoint_reader_context
        self._salient_keypoint_reader_context = None
        self._salient_keypoint_reader = None
        if context is not None:
            context.__exit__(None, None, None)

    def _create_image_datasets(self):
        """Create image datasets for high-res and low-res loading."""
        highres_conf = self.conf_highres
        self.highres_dataset = RomaVideoImageDataset(
            self.scene_parser.rgb_dir,
            ImageDatasetOptions(
                grayscale=highres_conf.grayscale,
                resize_max=highres_conf.resize_max,
                resize_force=highres_conf.resize_force,
                interpolation=highres_conf.interpolation,
            ),
            self.keyframe_sequence,
        )
        self.highres_dataset.normalize = False

        lowres_interpolation = highres_conf.interpolation
        self.lowres_dataset = RomaVideoImageDataset(
            self.scene_parser.rgb_dir,
            ImageDatasetOptions(
                resize_to_shape=(self.lowres_match_resolution, self.lowres_match_resolution),
                interpolation=lowres_interpolation,
            ),
            self.keyframe_sequence,
        )
        self.lowres_dataset.normalize = False

    def _infer_dense_match(self, name0, name1):
        """Infer one directed dense match with covariance on the CPU."""
        images0 = self._current_window_images[name0]
        images1 = self._current_window_images[name1]
        im_A_hr = images0["highres"].unsqueeze(0).cuda()
        im_A_lr = images0["lowres"].unsqueeze(0).cuda()
        im_B_hr = images1["highres"].unsqueeze(0).cuda()
        im_B_lr = images1["lowres"].unsqueeze(0).cuda()

        # Run matching
        from vidmap.utils.profiling import record_timing, sync_time

        _mt = sync_time()
        match = self.tracker_model.match_highres_pair(
            im_A_lr,
            im_B_lr,
            im_A_hr,
            im_B_hr,
            lowres_resolution=self.lowres_match_resolution,
            return_covariance=True,
        )
        if self._trackprop_first_match:
            record_timing("trackprop_first_match", sync_time() - _mt, first=True)
            self._trackprop_first_match = False

        return DenseMatchField(
            matches=match.matches[0].cpu().detach(),
            certainty=match.certainty[0].cpu().detach(),
            covariance=match.covariance[0].cpu().detach(),
            source_size=tuple(im_A_hr.shape[-2:][::-1]),
            target_size=tuple(im_B_hr.shape[-2:][::-1]),
        )

    def run(self):
        """Run streaming propagation while owning both asynchronous writers."""
        try:
            self._initialize_run()
            with (
                H5WriteQueue(
                    partial(write_pair_matches, match_path=self.paths.sparse_matches_path),
                ) as sparse_writer_queue,
                H5WriteQueue(
                    partial(write_pair_matches, match_path=self.extended_matches_path),
                ) as extended_writer_queue,
            ):
                return self._run_streaming_loop(sparse_writer_queue, extended_writer_queue)
        finally:
            self._cleanup()

    def _cleanup(self):
        """Release resources acquired by initialization or the streaming loop."""
        active_error = sys.exc_info()[0] is not None
        cleanup_error = None

        try:
            self._close_salient_keypoint_reader()
        except Exception as error:  # Preserve the processing failure while still releasing later resources.
            cleanup_error = error

        self._current_window_images.clear()
        for attribute in (
            "highres_dataset",
            "lowres_dataset",
            "state",
            "dense_hops",
        ):
            if attribute in vars(self):
                delattr(self, attribute)
        if cleanup_error is not None and not active_error:
            raise cleanup_error

    def _run_streaming_loop(
        self,
        sparse_writer_queue,
        extended_writer_queue,
    ):
        """Execute the numerical loop with writers owned by the caller."""

        self._open_salient_keypoint_reader()
        accepted_positions = [0]
        pending_lc_matches = ()
        pending_probe = None
        max_hop = max(self.conf.multiflow_hops)
        image_batches = iter(
            torch.utils.data.DataLoader(
                CandidateImageDataset(self.highres_dataset, self.lowres_dataset),
                batch_size=None,
                num_workers=self.conf.num_workers,
                prefetch_factor=None,
                pin_memory=True,
            )
        )
        loaded_position = -1

        for candidate_position in tqdm(
            range(1, len(self.keyframe_sequence)),
            desc=(
                "Admitting keyframes and building sparse tracks"
                if self.keyframe_options is not None
                else "Building sparse tracks (streaming)"
            ),
            mininterval=1.0,
            disable=not progress_bars_enabled(),
        ):
            anchor_position = accepted_positions[-1]
            anchor = self.keyframe_sequence[anchor_position]
            current = self.keyframe_sequence[candidate_position]
            lookahead_position = candidate_position + 1
            lookahead = (
                self.keyframe_sequence[lookahead_position]
                if lookahead_position < len(self.keyframe_sequence)
                else None
            )
            should_probe = (
                self.keyframe_options is not None
                and self.candidate_indices[candidate_position] not in self.forced_candidate_indices
                and lookahead is not None
            )
            accepted_names = tuple(self.keyframe_sequence[position] for position in accepted_positions[-max_hop:])
            records = build_multiflow_window(accepted_names, current, self.conf.multiflow_hops)
            retained_names = (*accepted_names, current)
            self._current_window_images = {
                name: self._current_window_images[name]
                for name in retained_names
                if name in self._current_window_images
            }
            required_position = lookahead_position if should_probe else candidate_position
            while loaded_position < required_position:
                highres, lowres = next(image_batches)
                loaded_position += 1
                name = self.keyframe_sequence[loaded_position]
                self._current_window_images[name] = {"highres": highres, "lowres": lowres}

            reusable_probe, pending_probe = pending_probe, None
            anchor_keypoints = self.get_keypoints(anchor)
            score = None
            if should_probe:
                probe_pair = (anchor, lookahead)
                probe = self._infer_dense_match(*probe_pair)
                pending_probe = (probe_pair, probe)
                anchor_index = self.candidate_indices[anchor_position]
                lookahead_index = self.candidate_indices[lookahead_position]
                score = score_keyframe_lookahead(
                    anchor_keypoints,
                    probe.matches,
                    probe.certainty,
                    source_grid_size=probe.source_size,
                    target_grid_size=probe.target_size,
                    source_original_size=get_image_size(self.scene_parser, anchor),
                    target_original_size=get_image_size(self.scene_parser, lookahead),
                    source_calibration=self.calibrations.get(anchor_index),
                    target_calibration=self.calibrations.get(lookahead_index),
                    certainty_threshold=self.keyframe_options.certainty_threshold,
                    max_normalized_keypoint_drift=self.keyframe_options.max_normalized_keypoint_drift,
                )
            skip_current = score is not None and score <= self.keyframe_options.target_frac
            if skip_current:
                continue

            fields = []
            for record in records:
                if reusable_probe is not None and record.pair == reusable_probe[0]:
                    fields.append(reusable_probe[1])
                else:
                    fields.append(self._infer_dense_match(*record.pair))
            fields = tuple(fields)
            direct = next(field for record, field in zip(records, fields) if record.hop == 1)
            if not hasattr(self, "state"):
                if direct.source_size != direct.target_size:
                    raise RuntimeError("Sparse propagation requires equal source and target processing grids")
                self.state = SparseTrackState(
                    self.conf,
                    original_size=get_image_size(self.scene_parser, anchor),
                    current_size=direct.source_size,
                    scheduled_hops=self.conf.multiflow_hops,
                )
                self.dense_hops = DenseHopBuffers.create(max(self.conf.multiflow_hops), direct.source_size)
            state = self.state
            dense_hops = self.dense_hops
            if any(field.source_size != state.current_size for field in fields):
                raise RuntimeError("Dense multiflow fields changed processing-grid size within one run")
            for record, field in zip(records, fields):
                dense_hops.matches[record.slot] = field.matches
                dense_hops.certainty[record.slot] = field.certainty
                dense_hops.covariance[record.slot] = field.covariance

            certainty = direct.certainty.clone()
            matches = direct.matches
            certainty[
                ~(
                    (matches[..., 0] >= 0)
                    & (matches[..., 0] < (state.current_width - 1))
                    & (matches[..., 1] >= 0)
                    & (matches[..., 1] < (state.current_height - 1))
                )
            ] = 0
            transition = state.advance(
                kf_id=len(accepted_positions) - 1,
                matches=direct.matches,
                certainty=certainty.to(state.device),
                covariance=direct.covariance,
                salient_keypoints=scale_keypoints(anchor_keypoints, state.scale_ratio),
                dense_hops=dense_hops,
            )

            save_keypoints(
                {
                    "name": [anchor],
                    "keypoints": transition.source_keypoints.astype(np.float32),
                    "scores": np.ones_like(transition.source_keypoints[0, :, 0], dtype=np.float32),
                    "image_size": state.original_size,
                    "uncertainty": transition.source_uncertainty,
                },
                self.paths.sparse_features_path,
            )

            if pending_lc_matches:
                flush_sequential_matches(
                    pending_lc_matches,
                    scene_parser=self.scene_parser,
                    sparse_features_path=self.paths.sparse_features_path,
                    match_threshold=self.lc_match_thresh,
                    writer_queue=extended_writer_queue,
                )

            pending_lc_matches = tuple((record.pair, field) for record, field in zip(records, fields))

            sparse_writer_queue.put(
                (
                    names_to_pair(anchor, current),
                    {
                        "matches0": torch.tensor(transition.matches)[None],
                        "matching_scores0": torch.tensor(transition.matching_scores)[None],
                    },
                )
            )
            accepted_positions.append(candidate_position)

        last_frame_kps, last_frame_covar = self.state.serialize_current_target()
        save_keypoints(
            {
                "name": [self.keyframe_sequence[accepted_positions[-1]]],
                "keypoints": last_frame_kps.astype(np.float32),
                "scores": np.ones_like(last_frame_kps[0, :, 0], dtype=np.float32),
                "image_size": self.state.original_size,
                "uncertainty": last_frame_covar,
            },
            self.paths.sparse_features_path,
        )
        flush_sequential_matches(
            pending_lc_matches,
            scene_parser=self.scene_parser,
            sparse_features_path=self.paths.sparse_features_path,
            match_threshold=self.lc_match_thresh,
            writer_queue=extended_writer_queue,
        )
        return tuple(self.candidate_indices[position] for position in accepted_positions)

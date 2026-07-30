"""
Streaming track frontend with composed dense matching and sparse propagation.

This module implements a streaming variant of track propagation that:
1. Computes dense matches on-the-fly as needed
2. Runs track propagation for consecutive pairs
3. Reuses the same dense fields for sequential LC matches
4. Outputs sparse features, propagated matches, and sequential LC matches

The streaming owner manages RoMa, image prefetch, H5 I/O, and dense-field reuse.
``SparseTrackState`` owns only the numerical propagation state.
"""

import logging
import sys
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vidmap.frontend.h5_write_queue import H5WriteQueue, save_keypoints, write_pair_matches
from vidmap.frontend.image_dataset import ImageDatasetOptions, get_image_size
from vidmap.frontend.paths import FrontendPaths
from vidmap.frontend.tracking.kernels import select_lc_matches_from_dense
from vidmap.frontend.tracking.multiflow import build_multiflow_iteration_plan
from vidmap.frontend.tracking.state import SparseTrackState
from vidmap.frontend.tracking.window import IterationWindowDataset, LRUCache, collate_iteration_window
from vidmap.frontend.video_images import RomaVideoImageDataset
from vidmap.utils.io import H5KeypointReader, get_keypoints
from vidmap.utils.keypoint_scaling import scale_keypoints
from vidmap.utils.logging import progress_bars_enabled
from vidmap.utils.parsers import names_to_pair

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DenseMatchField:
    matches: torch.Tensor
    certainty: torch.Tensor
    covariance: torch.Tensor


@dataclass
class DenseHopBuffers:
    matches: np.ndarray
    certainty: np.ndarray
    covariance: np.ndarray

    @classmethod
    def create(cls, window, current_size):
        width, height = current_size
        return cls(
            matches=np.full((window, height, width, 2), -1, dtype=np.float32),
            certainty=np.full((window, height, width), -1, dtype=np.float32),
            covariance=np.full((window, height, width, 2, 2), -1, dtype=np.float32),
        )

    def reset(self):
        self.matches.fill(-1)
        self.certainty.fill(-1)
        self.covariance.fill(-1)


def _write_sequential_match(
    name0,
    name1,
    dense,
    *,
    pair_names,
    scene_parser,
    sparse_features_path,
    match_threshold,
    writer_queue,
):
    """Convert one reused dense field into a queued sequential sparse match."""
    pair_name = names_to_pair(name0, name1)
    if pair_name not in pair_names:
        return
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


def _flush_sequential_matches(
    pairs,
    *,
    dense_lru,
    pair_names,
    scene_parser,
    sparse_features_path,
    match_threshold,
    writer_queue,
):
    for name0, name1 in pairs:
        _write_sequential_match(
            name0,
            name1,
            dense_lru[names_to_pair(name0, name1)],
            pair_names=pair_names,
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
        sequential_pairs=(),
    ):
        """Store dependencies; resource acquisition starts in :meth:`run`."""
        self.conf = conf
        self.tracker_model = tracker_model
        self.conf_highres = conf_highres
        self.lowres_match_resolution = int(lowres_match_resolution)
        self.extended_matches_path = extended_matches_path
        self.lc_match_thresh = lc_match_thresh
        self.sequential_pairs = tuple(sequential_pairs)
        self.scene_parser = scene_parser
        self.paths = paths
        self.keyframe_sequence = keyframe_sequence
        self._trackprop_first_match = True
        self._dense_lru = None
        self._current_window_images = {}
        self._salient_keypoint_reader = None
        self._salient_keypoint_reader_context = None

    def _initialize_run(self):
        self.sequential_pair_names = {names_to_pair(*pair) for pair in self.sequential_pairs}
        self.iteration_plan = build_multiflow_iteration_plan(
            self.keyframe_sequence,
            int(self.conf.window),
            self.conf.multiflow_hops,
        )

        # LRU reuse buffer for dense matches. Parent _init computes pair0 to
        # discover match resolution, so size this before super() to keep that
        # warmup match for the first real iteration.
        self._dense_lru = LRUCache(maxsize=3 * self.conf.window)

        # Image cache for current iteration: {image_idx: {"highres": tensor, "lowres": tensor}}
        self._current_window_images = {}
        self._salient_keypoint_reader = None
        self._salient_keypoint_reader_context = None

        self._create_image_datasets()
        first_pair = (self.keyframe_sequence[0], self.keyframe_sequence[1])
        first_dense = self._compute_dense_match(*first_pair)
        original_size = get_image_size(self.scene_parser, self.keyframe_sequence[0])
        current_size = first_dense.matches.shape[:2][::-1]
        self.state = SparseTrackState(
            self.conf,
            original_size=original_size,
            current_size=current_size,
            scheduled_hops=self.iteration_plan.hops,
        )
        self.dense_hops = DenseHopBuffers.create(self.conf.window, current_size)

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

        self.name_to_idx = {name: i for i, name in enumerate(self.keyframe_sequence)}

    def _compute_dense_match(self, name0, name1):
        """Compute dense match for a single pair."""
        pair_name = names_to_pair(name0, name1)

        # Load images (prefer cache if available, set by DataLoader)
        idx0 = self.name_to_idx[name0]
        idx1 = self.name_to_idx[name1]

        if idx0 in self._current_window_images:
            im_A_hr = self._current_window_images[idx0]["highres"].unsqueeze(0).cuda()
            im_A_lr = self._current_window_images[idx0]["lowres"].unsqueeze(0).cuda()
        else:
            im_A_hr = self.highres_dataset[idx0]["image"].unsqueeze(0).cuda()
            im_A_lr = self.lowres_dataset[idx0]["image"].unsqueeze(0).cuda()

        if idx1 in self._current_window_images:
            im_B_hr = self._current_window_images[idx1]["highres"].unsqueeze(0).cuda()
            im_B_lr = self._current_window_images[idx1]["lowres"].unsqueeze(0).cuda()
        else:
            im_B_hr = self.highres_dataset[idx1]["image"].unsqueeze(0).cuda()
            im_B_lr = self.lowres_dataset[idx1]["image"].unsqueeze(0).cuda()

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

        result = DenseMatchField(
            matches=match.matches[0].cpu().detach(),
            certainty=match.certainty[0].cpu().detach(),
            covariance=match.covariance[0].cpu().detach(),
        )

        # Store in LRU reuse buffer
        self._dense_lru[pair_name] = result

        return result

    def get_matches(self, pair):
        """Get a dense field from the in-memory LRU or compute it on demand."""
        name0, name1 = pair
        pair_name = names_to_pair(name0, name1)

        # Check in-memory reuse buffer
        if pair_name in self._dense_lru:
            return self._dense_lru[pair_name]

        return self._compute_dense_match(name0, name1)

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
        if self._dense_lru is not None:
            self._dense_lru.clear()
        for attribute in (
            "highres_dataset",
            "lowres_dataset",
            "state",
            "dense_hops",
            "iteration_plan",
            "sequential_pair_names",
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
        state = self.state
        dense_hops = self.dense_hops
        pending_lc_pairs = ()

        # Create DataLoader for iteration windows.
        window_dataset = IterationWindowDataset(
            self.highres_dataset,
            self.lowres_dataset,
            self.iteration_plan.windows,
        )
        window_loader = DataLoader(
            window_dataset,
            batch_size=1,
            num_workers=self.conf.num_workers,
            prefetch_factor=None,
            collate_fn=collate_iteration_window,
            pin_memory=True,
        )
        for batch in tqdm(
            window_loader,
            desc="Building sparse tracks (streaming)",
            mininterval=1.0,
            disable=not progress_bars_enabled(),
        ):
            kf_id = batch["kf_id"]

            self._current_window_images = batch["images"]

            pair = (
                self.keyframe_sequence[kf_id],
                self.keyframe_sequence[kf_id + 1],
            )
            dense = self.get_matches(pair)
            dmatches = dense.matches
            dcertainty = dense.certainty
            dcovs = dense.covariance

            salient_kps = self.get_keypoints(self.keyframe_sequence[kf_id])
            salient_kps = scale_keypoints(salient_kps, state.scale_ratio)

            # NOTE: COV_SCALE is NOT applied here - raw covariances are propagated and saved
            # COV_SCALE is applied only at filtering/refinement time to maintain thresholds
            # Set certainty for matches outside of canvas to 0
            # Clone here (not in get_matches) to avoid cloning for window pairs that don't modify
            dcertainty = dcertainty.clone()
            filter_mask = ~(
                (dmatches[..., 0] >= 0)
                & (dmatches[..., 0] < (state.current_width - 1))
                & (dmatches[..., 1] >= 0)
                & (dmatches[..., 1] < (state.current_height - 1))
            )
            dcertainty[filter_mask] = 0
            dcertainty = dcertainty.to(state.device)

            # Build scheduled source-to-current dense fields for long-track filtering.
            prev_records = self.iteration_plan.windows[kf_id]
            dense_hops.reset()
            for record in prev_records:
                prev_pair = record.pair
                dense = self.get_matches(prev_pair)
                dense_hops.matches[record.slot] = dense.matches
                dense_hops.certainty[record.slot] = dense.certainty
                dense_hops.covariance[record.slot] = dense.covariance

            transition = state.advance(
                kf_id=kf_id,
                matches=dmatches,
                certainty=dcertainty,
                covariance=dcovs,
                salient_keypoints=salient_kps,
                dense_hops=dense_hops,
            )

            save_keypoints(
                {
                    "name": [self.keyframe_sequence[kf_id]],
                    "keypoints": transition.source_keypoints.astype(np.float32),
                    "scores": np.ones_like(transition.source_keypoints[0, :, 0], dtype=np.float32),
                    "image_size": state.original_size,
                    "uncertainty": transition.source_uncertainty,
                },
                self.paths.sparse_features_path,
            )

            if pending_lc_pairs:
                _flush_sequential_matches(
                    pending_lc_pairs,
                    dense_lru=self._dense_lru,
                    pair_names=self.sequential_pair_names,
                    scene_parser=self.scene_parser,
                    sparse_features_path=self.paths.sparse_features_path,
                    match_threshold=self.lc_match_thresh,
                    writer_queue=extended_writer_queue,
                )

            pending_lc_pairs = tuple(record.pair for record in prev_records)

            pair_name = names_to_pair(self.keyframe_sequence[kf_id], self.keyframe_sequence[kf_id + 1])
            sparse_writer_queue.put(
                (
                    pair_name,
                    {
                        "matches0": torch.tensor(transition.matches)[None],
                        "matching_scores0": torch.tensor(transition.matching_scores)[None],
                    },
                )
            )

            if kf_id == len(self.keyframe_sequence) - 2:
                last_frame_kps, last_frame_covar = state.serialize_current_target()

                save_keypoints(
                    {
                        "name": [self.keyframe_sequence[kf_id + 1]],
                        "keypoints": last_frame_kps.astype(np.float32),
                        "scores": np.ones_like(last_frame_kps[0, :, 0], dtype=np.float32),
                        "image_size": state.original_size,
                        "uncertainty": last_frame_covar,
                    },
                    self.paths.sparse_features_path,
                )

                _flush_sequential_matches(
                    pending_lc_pairs,
                    dense_lru=self._dense_lru,
                    pair_names=self.sequential_pair_names,
                    scene_parser=self.scene_parser,
                    sparse_features_path=self.paths.sparse_features_path,
                    match_threshold=self.lc_match_thresh,
                    writer_queue=extended_writer_queue,
                )

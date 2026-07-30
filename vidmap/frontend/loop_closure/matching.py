"""
Extended-match frontend, cache repair, loading, and tcorr merging.

This module computes LC (loop closure) matches for retrieval pairs by:
1. Computing dense matches on-the-fly
2. Sampling at sparse keypoint locations
3. Appending to the extended-match artifact

Uses DataLoader for image prefetching, matching the streaming track propagator.
"""

import logging
from functools import partial

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vidmap.frontend.cache import CacheMetadataMismatch, inspect_incremental_items
from vidmap.frontend.correspondences import canonical_pair_names
from vidmap.frontend.h5_write_queue import H5WriteQueue, write_pair_matches
from vidmap.frontend.image_dataset import ImageDatasetOptions, get_image_size
from vidmap.frontend.tracking.kernels import select_lc_matches_from_dense
from vidmap.frontend.video_images import RomaVideoImageDataset, load_roma_resolution_pair
from vidmap.utils.io import H5KeypointReader, ordered_pair_images
from vidmap.utils.logging import progress_bars_enabled
from vidmap.utils.parsers import names_to_pair

logger = logging.getLogger(__name__)


class RetrievalPairDataset(Dataset):
    """Dataset that loads image pairs for retrieval LC frontend."""

    def __init__(
        self,
        highres_dataset,
        lowres_dataset,
        retrieval_pairs,
        name_to_idx,
    ):
        """
        Args:
            highres_dataset: RomaVideoImageDataset for high-res images
            lowres_dataset: RomaVideoImageDataset for low-res images
            retrieval_pairs: List of (name0, name1) pairs
            name_to_idx: Dict mapping image names to dataset indices
        """
        self.highres_dataset = highres_dataset
        self.lowres_dataset = lowres_dataset
        self.retrieval_pairs = retrieval_pairs
        self.name_to_idx = name_to_idx

    def __len__(self):
        return len(self.retrieval_pairs)

    def __getitem__(self, idx):
        """Load images for a pair."""
        name0, name1 = self.retrieval_pairs[idx]
        idx0, idx1 = self.name_to_idx[name0], self.name_to_idx[name1]
        image0_highres, image0_lowres = load_roma_resolution_pair(
            self.highres_dataset,
            self.lowres_dataset,
            idx0,
        )
        image1_highres, image1_lowres = load_roma_resolution_pair(
            self.highres_dataset,
            self.lowres_dataset,
            idx1,
        )

        result = {
            "pair_idx": idx,
            "name0": name0,
            "name1": name1,
            "im_A_hr": image0_highres["image"],
            "im_B_hr": image1_highres["image"],
            "im_A_lr": image0_lowres["image"],
            "im_B_lr": image1_lowres["image"],
        }

        return result


def collate_retrieval_pair(batch):
    """Collate function - pass through since batch_size=1."""
    return batch[0]


def _match_loop_closures_streaming(
    scene_parser,
    sparse_features_path,
    tracker_model,
    retrieval_pairs,
    conf_highres,
    lowres_match_resolution,
    lc_match_thresh=0.5,
    lc_writer_queue=None,
):
    """
    Streaming LC frontend for retrieval pairs.

    Computes dense matches on-the-fly and selects sparse LC matches,
    appending them to the extended-match artifact.

    Args:
        scene_parser: Scene parser instance
        sparse_features_path: Sparse feature artifact
        tracker_model: Tracker model instance
        retrieval_pairs: List of (name0, name1) pairs from retrieval
        conf_highres: High-res frontend config
        lc_match_thresh: Certainty threshold for LC matches
    """
    if not retrieval_pairs:
        return

    # Collect unique images from retrieval pairs
    retrieval_images = ordered_pair_images(retrieval_pairs)

    # Create image datasets
    highres_conf = conf_highres
    highres_dataset = RomaVideoImageDataset(
        scene_parser.rgb_dir,
        ImageDatasetOptions(
            grayscale=highres_conf.grayscale,
            resize_max=highres_conf.resize_max,
            resize_force=highres_conf.resize_force,
            interpolation=highres_conf.interpolation,
        ),
        retrieval_images,
    )
    highres_dataset.normalize = False

    lowres_interpolation = highres_conf.interpolation
    _lr_res = int(lowres_match_resolution)
    lowres_dataset = RomaVideoImageDataset(
        scene_parser.rgb_dir,
        ImageDatasetOptions(resize_to_shape=(_lr_res, _lr_res), interpolation=lowres_interpolation),
        retrieval_images,
    )
    lowres_dataset.normalize = False

    name_to_idx = {name: i for i, name in enumerate(retrieval_images)}

    # Create DataLoader for pair prefetching
    pair_dataset = RetrievalPairDataset(
        highres_dataset,
        lowres_dataset,
        retrieval_pairs,
        name_to_idx,
    )
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=collate_retrieval_pair,
        pin_memory=True,
    )

    # Stats for logging
    _stats = {"computed": 0, "total_matches": 0}
    with H5KeypointReader(sparse_features_path, max_size=512) as keypoint_reader:
        for batch in tqdm(
            pair_loader,
            desc="Streaming LC frontend for retrieval pairs",
            disable=not progress_bars_enabled(),
        ):
            name0, name1 = batch["name0"], batch["name1"]
            pair_name = names_to_pair(name0, name1)

            _stats["computed"] += 1

            # Use prefetched images from DataLoader
            im_A_hr = batch["im_A_hr"].unsqueeze(0).cuda()
            im_B_hr = batch["im_B_hr"].unsqueeze(0).cuda()
            im_A_lr = batch["im_A_lr"].unsqueeze(0).cuda()
            im_B_lr = batch["im_B_lr"].unsqueeze(0).cuda()

            from vidmap.utils.profiling import record_timing, sync_time

            _mt = sync_time()
            match = tracker_model.match_highres_pair(
                im_A_lr,
                im_B_lr,
                im_A_hr,
                im_B_hr,
                lowres_resolution=_lr_res,
            )
            record_timing("lc_first_match", sync_time() - _mt, first=True)
            # Free input images immediately
            del im_A_hr, im_B_hr, im_A_lr, im_B_lr

            matches_dense = match.matches[0]
            certainty = match.certainty[0]
            del match

            # Skip pair if network produced NaN (numerical instability on hard pairs)
            if not torch.all(torch.isfinite(matches_dense)):
                logger.warning(f"[LC Streaming] NaN in dense warp for {pair_name}, skipping pair")
                continue

            # Move to CPU for processing
            matches_dense = matches_dense.cpu()
            certainty = certainty.cpu()

            # Load sparse keypoints
            kpts0 = keypoint_reader.get(name0)
            kpts1 = keypoint_reader.get(name1)

            # Get dimensions
            W0, H0 = get_image_size(scene_parser, name0)
            W1, H1 = get_image_size(scene_parser, name1)

            # === Core LC frontend ===
            lc_result = select_lc_matches_from_dense(
                kpts0,
                kpts1,
                matches_dense,
                certainty,
                source_size=(W0, H0),
                target_size=(W1, H1),
                lc_match_thresh=lc_match_thresh,
            )
            matches0 = lc_result["matches0"]
            matching_scores0 = lc_result["matching_scores0"]

            num_matches = (matches0 >= 0).sum()
            _stats["total_matches"] += num_matches

            # Also write to disk if caching is enabled
            if lc_writer_queue is not None:
                lc_writer_queue.put(
                    (
                        pair_name,
                        {
                            "matches0": torch.from_numpy(matches0)[None],
                            "matching_scores0": torch.from_numpy(matching_scores0).float()[None],
                        },
                    )
                )

            # Free tensors from this iteration
            del matches_dense, certainty

    # Log summary
    logger.info(f"[LC Streaming] Dense matches computed: {_stats['computed']}")
    logger.info(f"[LC Streaming] Total sparse matches selected: {_stats['total_matches']}")


def match_loop_closures_streaming(
    scene_parser,
    sparse_features_path,
    extended_matches_path,
    tracker_model,
    retrieval_pairs,
    conf_highres,
    lowres_match_resolution,
    lc_match_thresh=0.5,
):
    """Run LC frontend with deterministic writer teardown on every exit."""
    if not retrieval_pairs:
        return
    with H5WriteQueue(partial(write_pair_matches, match_path=extended_matches_path)) as writer_queue:
        return _match_loop_closures_streaming(
            scene_parser,
            sparse_features_path,
            tracker_model,
            retrieval_pairs,
            conf_highres,
            lowres_match_resolution,
            lc_match_thresh,
            writer_queue,
        )


def repair_extended_match_pairs(
    pairs,
    *,
    label,
    paths,
    metadata,
    tracker,
    scene_parser,
    conf_highres,
    lowres_match_resolution,
    lc_match_thresh,
):
    """Repair and verify one planned set of extended-match pairs."""
    expected_names = canonical_pair_names(pairs)
    present_names, missing_names = inspect_incremental_items(
        paths.extended_matches_path,
        expected_names,
        metadata,
        repair_malformed=True,
    )
    missing_name_set = set(missing_names)
    missing_pairs = [pair for pair in pairs if names_to_pair(*pair) in missing_name_set]
    if missing_pairs:
        logger.info("Streaming LC frontend for %d %s pairs", len(missing_pairs), label)
        match_loop_closures_streaming(
            scene_parser=scene_parser,
            sparse_features_path=paths.sparse_features_path,
            extended_matches_path=paths.extended_matches_path,
            tracker_model=tracker.get(),
            retrieval_pairs=missing_pairs,
            conf_highres=conf_highres,
            lowres_match_resolution=lowres_match_resolution,
            lc_match_thresh=lc_match_thresh,
        )
        _, still_missing = inspect_incremental_items(
            paths.extended_matches_path,
            expected_names,
            metadata,
            repair_malformed=True,
        )
        if still_missing:
            raise CacheMetadataMismatch(
                f"{paths.extended_matches_path}: {label} LC repair omitted " f"{len(still_missing)} planned pairs"
            )
    elif pairs:
        logger.info("Skipping %s LC frontend; all %d pairs exist", label, len(pairs))
    return len(present_names), len(missing_pairs)


def _filtered_matches_from_sparse_arrays(raw_matches, raw_scores, reverse, lc_match_thresh):
    valid_idx = np.where(raw_matches != -1)[0]
    matches = np.stack([valid_idx, raw_matches[valid_idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = raw_scores[valid_idx]
    return matches[scores > lc_match_thresh]


def _cached_extended_matches_from_open_file(extended_matches_file, pair, lc_match_thresh):
    pair_name_fwd = names_to_pair(pair[0], pair[1])
    if pair_name_fwd in extended_matches_file:
        stored_pair = pair_name_fwd
        reverse = False
    else:
        pair_name_rev = names_to_pair(pair[1], pair[0])
        if pair_name_rev not in extended_matches_file:
            return np.empty((0, 2), dtype=np.int64)
        stored_pair = pair_name_rev
        reverse = True
    raw_matches = extended_matches_file[stored_pair]["matches0"].__array__()
    raw_scores = extended_matches_file[stored_pair]["matching_scores0"].__array__()
    return _filtered_matches_from_sparse_arrays(raw_matches, raw_scores, reverse, lc_match_thresh)


def collect_cached_extended_matches(*, extended_pairs, extended_matches_path, lc_match_thresh):
    """Collect extended matches while keeping cached HDF5 matches open once."""
    matches = {}
    with h5py.File(str(extended_matches_path), "r", libver="latest") as open_matches:
        for pair in extended_pairs:
            filtered_matches = _cached_extended_matches_from_open_file(open_matches, pair, lc_match_thresh)
            if len(filtered_matches) > 0:
                matches[pair] = filtered_matches
    return matches


def merge_matches(tcorr: dict, matches: dict):
    """Merge extended matches into transitive correspondences and build LC masks."""
    undirected = {}
    for pair in tcorr:
        key = frozenset(pair)
        if key in undirected:
            raise ValueError(f"Duplicate undirected transitive pair: {undirected[key]!r} and {pair!r}")
        undirected[key] = pair
    lc_masks = {}
    for (name0, name1), match_array in matches.items():
        pair = (name0, name1)
        existing_pair = undirected.get(frozenset(pair), pair)
        mask_pair = pair
        if existing_pair != pair:
            match_array = np.flip(match_array, axis=1)
            mask_pair = existing_pair
        if existing_pair in tcorr:
            mask0 = ~np.isin(match_array[:, 0], tcorr[existing_pair][:, 0])
            mask1 = ~np.isin(match_array[:, 1], tcorr[existing_pair][:, 1])
            mask = mask0 & mask1
            tcorr[existing_pair] = np.vstack([tcorr[existing_pair], match_array[mask]])
        else:
            tcorr[existing_pair] = match_array
            undirected[frozenset(existing_pair)] = existing_pair
            mask = np.ones(len(match_array), dtype=bool)
        lc_masks[mask_pair] = np.zeros(len(tcorr[existing_pair]), dtype=bool)
        if mask.sum() > 0:
            lc_masks[mask_pair][-mask.sum() :] = True
    for pair in tcorr:
        if pair not in lc_masks:
            lc_masks[pair] = np.zeros(len(tcorr[pair]), dtype=bool)
    return tcorr, lc_masks

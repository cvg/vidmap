"""MegaLoc descriptor frontend and retrieval-pair generation."""

import logging
import pprint
from collections import defaultdict
from itertools import chain
from pathlib import Path

import h5py
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

from vidmap.frontend.cache import (
    cache_metadata,
    inspect_incremental_items,
    mark_incremental_cache_complete,
    prepare_incremental_cache,
)
from vidmap.frontend.image_dataset import ImageDataset, ImageDatasetOptions
from vidmap.frontend.models.megaloc import MegaLocDescriptorModel, megaloc_cache_identity
from vidmap.utils.device import empty_device_cache
from vidmap.utils.logging import progress_bars_enabled

RETRIEVAL_PAIR_SELECTION_POLICY_VERSION = 3

logger = logging.getLogger(__name__)


def _retrieval_config() -> dict:
    return {
        "model": {},
        "preprocessing": {
            "resize_max": 1024,
            "resize_force": True,
        },
    }


def retrieval_cache_identity(image_list, image_content_fingerprint):
    retrieval_conf = _retrieval_config()
    return cache_metadata(
        stage="retrieval_features",
        config={
            "model": megaloc_cache_identity(),
            "preprocessing": retrieval_conf["preprocessing"],
        },
        ordered_inputs={
            "images": image_list,
            "image_content": image_content_fingerprint,
        },
        payload_format="per-image-global-descriptors",
    )


@torch.no_grad()
def compute_retrieval_features(
    scene_parser,
    retrieval_features_path,
    image_list,
    cache_identity,
    overwrite,
    *,
    device: torch.device,
):
    """
    Compute retrieval features for image matching.

    Args:
        scene_parser: Scene parser with rgb_dir
        retrieval_features_path: Retrieval descriptor artifact
        image_list: List of image names to process
        cache_identity: Cache fingerprint
        overwrite: Overwrite existing cache
        device: Target execution device

    """
    retrieval_conf = _retrieval_config()
    if cache_identity is None:
        raise TypeError("cache_identity is required")
    identity = cache_identity

    logger.debug("Retrieval feature configuration:\n%s", pprint.pformat(retrieval_conf))

    dataset = ImageDataset(
        scene_parser.rgb_dir,
        ImageDatasetOptions(**retrieval_conf["preprocessing"]),
        image_list,
    )
    retrieval_features_path = Path(retrieval_features_path)
    retrieval_features_path.parent.mkdir(parents=True, exist_ok=True)
    expected_names = list(dataset.names)
    prepare_incremental_cache(retrieval_features_path, identity, overwrite=overwrite)
    present_names, _ = inspect_incremental_items(
        retrieval_features_path,
        expected_names,
        identity,
        repair_malformed=True,
    )
    skip_names = set(present_names)

    dataset.names = [name for name in dataset.names if name not in skip_names]
    if len(dataset.names) == 0:
        logger.info("Skipping retrieval frontend because every item is cached")
        mark_incremental_cache_complete(retrieval_features_path, identity, expected_names)
        empty_device_cache(device)
        logger.info("Retrieval features available at %s", retrieval_features_path)
        return

    model = MegaLocDescriptorModel().eval().to(device)
    loader = torch.utils.data.DataLoader(dataset, num_workers=1, shuffle=False, pin_memory=(device.type == "cuda"))
    for idx, data in enumerate(tqdm(loader, disable=not progress_bars_enabled())):
        name = dataset.names[idx]
        pred = model({"image": data["image"].to(device, non_blocking=True)})
        pred = {key: value[0].cpu().numpy() for key, value in pred.items()}
        pred["image_size"] = data["original_size"][0].numpy()

        for key in pred:
            if pred[key].dtype == np.float32:
                pred[key] = pred[key].astype(np.float16)

        with h5py.File(str(retrieval_features_path), "a", libver="latest") as fd:
            try:
                if name in fd:
                    del fd[name]
                group = fd.create_group(name)
                for key, value in pred.items():
                    group.create_dataset(key, data=value)
            except OSError as error:
                if "No space left on device" in error.args[0]:
                    logger.error("Out of disk space while storing retrieval descriptors")
                    del group, fd[name]
                raise error

        del pred

    mark_incremental_cache_complete(retrieval_features_path, identity, expected_names)
    del model

    empty_device_cache(device)

    logger.info("Retrieval features saved to %s", retrieval_features_path)


def _load_descriptors(names, hfile):
    """Load one ordered descriptor set."""
    descriptors = np.stack([hfile[name]["global_descriptor"][()] for name in names])
    return torch.as_tensor(descriptors, dtype=torch.float)


def _pairs_from_score_matrix(scores, invalid, num_select, min_score, return_scores):
    invalid = torch.as_tensor(invalid, device=scores.device)
    invalid |= scores < min_score
    scores.masked_fill_(invalid, float("-inf"))
    count = min(num_select, scores.shape[1])
    indices_tensor = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :count]
    values_tensor = torch.gather(scores, 1, indices_tensor)
    indices = indices_tensor.cpu().numpy()
    values = values_tensor.cpu().numpy()
    valid = values_tensor.isfinite().cpu().numpy()
    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j], float(values[i, j])) if return_scores else (i, indices[i, j]))
    return pairs


def _retrieve_pairs(
    path, names, reference_query_pairs, num_matched, min_score, return_scores, *, device: torch.device
):
    with h5py.File(str(path), "r", libver="latest") as hfile:
        available = [name for name in dict.fromkeys(names) if name in hfile and "global_descriptor" in hfile[name]]
        if not available:
            return []
        descriptors = _load_descriptors(available, hfile).to(device)
    indices = {name: index for index, name in enumerate(available)}
    pairs = []
    for reference, candidates in reference_query_pairs:
        if reference not in indices:
            continue
        valid_queries = [query for query in candidates if query in indices]
        if not valid_queries:
            continue
        reference_descriptor = descriptors[indices[reference] : indices[reference] + 1]
        candidate_descriptors = descriptors[[indices[query] for query in valid_queries]]
        similarities = torch.einsum("id,jd->ij", reference_descriptor, candidate_descriptors)
        self_mask = np.array([reference])[:, None] == np.array(valid_queries)[None]
        pair_indices = _pairs_from_score_matrix(
            similarities,
            self_mask,
            num_matched,
            min_score,
            return_scores,
        )
        for pair in pair_indices:
            if return_scores:
                _, query_index, score = pair
                pairs.append((reference, valid_queries[query_index], score))
            else:
                _, query_index = pair
                pairs.append((reference, valid_queries[query_index]))
    return pairs


def generate_retrieval_pairs(
    sequence,
    tcorr_counts,
    sequential_pairs,
    retrieval_path,
    tcorr_min_matches,
    retrieval_min_score,
    nquery,
    lc_pair_nms=False,
    lc_pair_nms_radius=2,
    max_pairs_per_frame_pair: int | None = None,
    *,
    frame_camera_indices,
    device: torch.device,
):
    """Generate retrieval pairs excluding sequential and sufficiently tracked pairs."""
    if not retrieval_path.exists():
        raise ValueError("Retrieval features not found. Run compute_retrieval_features step first.")

    excluded_images = defaultdict(set)
    for first, second in chain(
        sequential_pairs, (pair for pair, count in tcorr_counts.items() if count > tcorr_min_matches)
    ):
        excluded_images[first].add(second)
        excluded_images[second].add(first)

    def candidates():
        for reference in sequence:
            excluded = excluded_images[reference]
            frame = frame_camera_indices[reference][0]
            queries = [name for name in sequence if name not in excluded and frame_camera_indices[name][0] != frame]
            if queries:
                yield reference, queries

    reference_queries = iter(candidates())
    first = next(reference_queries, None)
    if first is None:
        logger.info("No untracked pairs found for retrieval")
        return []

    limit_frame_pairs = max_pairs_per_frame_pair is not None
    retrieval_pairs = _retrieve_pairs(
        retrieval_path,
        sequence,
        chain((first,), reference_queries),
        nquery,
        retrieval_min_score,
        lc_pair_nms or limit_frame_pairs,
        device=device,
    )
    if limit_frame_pairs:
        retrieval_pairs = _limit_rig_frame_pairs(retrieval_pairs, frame_camera_indices, max_pairs_per_frame_pair)
    if lc_pair_nms:
        raw_count = len(retrieval_pairs)
        retrieval_pairs = endpoint_nms_retrieval_pairs(
            retrieval_pairs, sequence, radius=lc_pair_nms_radius, frame_camera_indices=frame_camera_indices
        )
        logger.info(
            "LC pair endpoint NMS radius=%d: %d -> %d pairs",
            lc_pair_nms_radius,
            raw_count,
            len(retrieval_pairs),
        )
    else:
        if limit_frame_pairs:
            retrieval_pairs = [(name0, name1) for name0, name1, _ in retrieval_pairs]
        retrieval_pairs = natsorted(tuple(natsorted(pair)) for pair in {frozenset(pair) for pair in retrieval_pairs})

    logger.info("Generated %d retrieval pairs, excluding sequential pairs", len(retrieval_pairs))
    return retrieval_pairs


def endpoint_nms_retrieval_pairs(scored_pairs, sequence, radius, frame_camera_indices):
    """Keep the best retrieval edge per local query/target neighborhood."""
    if radius < 0:
        raise ValueError(f"lc_pair_nms_radius must be non-negative, got {radius}")

    seq_index = {name: frame_camera_indices[name][0] for name in sequence}
    best_by_pair = {}
    for name0, name1, score in scored_pairs:
        if name0 not in seq_index or name1 not in seq_index or name0 == name1:
            continue
        idx0 = seq_index[name0]
        idx1 = seq_index[name1]
        if idx0 <= idx1:
            pair = (name0, name1)
            endpoints = (idx0, idx1)
        else:
            pair = (name1, name0)
            endpoints = (idx1, idx0)
        if pair not in best_by_pair or score > best_by_pair[pair][0]:
            best_by_pair[pair] = (score, endpoints)

    candidates = sorted(
        ((score, endpoints, pair) for pair, (score, endpoints) in best_by_pair.items()),
        key=lambda item: (-item[0], item[2][0], item[2][1]),
    )
    accepted = []
    accepted_endpoints = defaultdict(list)
    for _score, endpoints, pair in candidates:
        camera_pair = tuple(frame_camera_indices[name][1] for name in pair)
        suppress = any(
            abs(endpoints[0] - previous[0]) <= radius and abs(endpoints[1] - previous[1]) <= radius
            for previous in accepted_endpoints[camera_pair]
        )
        if suppress:
            continue
        accepted.append(pair)
        accepted_endpoints[camera_pair].append(endpoints)
    # Preserve temporal source/target order for inference; only persisted match
    # rows are canonicalized, after sampling the directed dense field.
    return natsorted(accepted)


def _limit_rig_frame_pairs(scored_pairs, frame_camera_indices, maximum):
    """Bound verification per exposure pair while keeping the strongest views."""
    accepted, seen, counts = [], set(), defaultdict(int)
    for name0, name1, score in sorted(scored_pairs, key=lambda item: (-item[2], item[0], item[1])):
        image_pair = frozenset((name0, name1))
        frame_pair = frozenset((frame_camera_indices[name0][0], frame_camera_indices[name1][0]))
        if image_pair in seen or counts[frame_pair] >= maximum:
            continue
        seen.add(image_pair)
        counts[frame_pair] += 1
        accepted.append((name0, name1, score))
    return accepted

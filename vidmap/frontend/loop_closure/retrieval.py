"""MegaLoc descriptor frontend and retrieval-pair generation."""

import logging
import pprint
from collections import defaultdict
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
from vidmap.utils.logging import progress_bars_enabled

RETRIEVAL_PAIR_SELECTION_POLICY_VERSION = 1

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
):
    """
    Compute retrieval features for image matching.

    Args:
        scene_parser: Scene parser with rgb_dir
        retrieval_features_path: Retrieval descriptor artifact
        image_list: List of image names to process

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
        torch.cuda.empty_cache()
        logger.info("Retrieval features available at %s", retrieval_features_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MegaLocDescriptorModel().eval().to(device)
    loader = torch.utils.data.DataLoader(dataset, num_workers=1, shuffle=False, pin_memory=True)
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

    torch.cuda.empty_cache()

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


def _retrieve_pairs(path, reference_query_dict, num_matched, min_score, return_scores):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with h5py.File(str(path), "r", libver="latest") as hfile:

        def has_descriptor(name):
            return name in hfile and "global_descriptor" in hfile[name]

        references = [name for name in reference_query_dict if has_descriptor(name)]
        queries = list(
            dict.fromkeys(
                query for candidates in reference_query_dict.values() for query in candidates if has_descriptor(query)
            )
        )
        if not references or not queries:
            return []
        reference_descriptors = _load_descriptors(references, hfile).to(device)
        query_descriptors = _load_descriptors(queries, hfile).to(device)
    reference_indices = {name: index for index, name in enumerate(references)}
    query_indices = {name: index for index, name in enumerate(queries)}
    pairs = []
    for reference, candidates in reference_query_dict.items():
        if reference not in reference_indices:
            continue
        valid_queries = [query for query in candidates if query in query_indices]
        if not valid_queries:
            continue
        reference_descriptor = reference_descriptors[reference_indices[reference] : reference_indices[reference] + 1]
        candidate_descriptors = query_descriptors[[query_indices[query] for query in valid_queries]]
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
    tcorr,
    sequential_pairs,
    retrieval_path,
    tcorr_min_matches,
    retrieval_min_score,
    nquery,
    lc_pair_nms=False,
    lc_pair_nms_radius=2,
):
    """Generate retrieval pairs excluding sequential and sufficiently tracked pairs."""
    if not retrieval_path.exists():
        raise ValueError("Retrieval features not found. Run compute_retrieval_features step first.")

    sequential_pairs_set = {frozenset(pair) for pair in sequential_pairs}
    tcorr_pairs_set = {frozenset(pair) for pair, matches in tcorr.items() if len(matches) > tcorr_min_matches}
    untracked_pairs = defaultdict(list)
    for id_a in range(len(sequence)):
        for id_b in range(len(sequence)):
            if id_a == id_b:
                continue
            pair = frozenset([sequence[id_a], sequence[id_b]])
            if pair not in tcorr_pairs_set and pair not in sequential_pairs_set:
                untracked_pairs[sequence[id_a]].append(sequence[id_b])

    if not untracked_pairs:
        logger.info("No untracked pairs found for retrieval")
        return []

    retrieval_pairs = _retrieve_pairs(
        retrieval_path,
        untracked_pairs,
        nquery,
        retrieval_min_score,
        lc_pair_nms,
    )
    if lc_pair_nms:
        raw_count = len(retrieval_pairs)
        retrieval_pairs = endpoint_nms_retrieval_pairs(retrieval_pairs, sequence, radius=lc_pair_nms_radius)
        logger.info(
            "LC pair endpoint NMS radius=%d: %d -> %d pairs",
            lc_pair_nms_radius,
            raw_count,
            len(retrieval_pairs),
        )
    else:
        retrieval_pairs = natsorted(tuple(natsorted(pair)) for pair in {frozenset(pair) for pair in retrieval_pairs})

    logger.info("Generated %d retrieval pairs, excluding sequential pairs", len(retrieval_pairs))
    return retrieval_pairs


def endpoint_nms_retrieval_pairs(scored_pairs, sequence, radius):
    """Keep the best retrieval edge per local query/target neighborhood."""
    if radius < 0:
        raise ValueError(f"lc_pair_nms_radius must be non-negative, got {radius}")

    seq_index = {name: idx for idx, name in enumerate(sequence)}
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
    accepted_endpoints = []
    for _score, endpoints, pair in candidates:
        suppress = any(
            abs(endpoints[0] - previous[0]) <= radius and abs(endpoints[1] - previous[1]) <= radius
            for previous in accepted_endpoints
        )
        if suppress:
            continue
        accepted.append(pair)
        accepted_endpoints.append(endpoints)
    return natsorted(accepted)

"""I/O functions for reading and writing data, inspired by Hierarchical-Localization."""

import os
from collections import OrderedDict
from pathlib import Path

import cv2
import h5py
import numpy as np

from vidmap.utils.parsers import names_to_pair


def drop_page_cache(path):
    fd = os.open(str(path), os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def read_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    drop_page_cache(path)
    return image


def _read_keypoints_from_group(grp, return_uncertainty: bool = False):
    dset = grp["keypoints"]
    p = dset.__array__()
    if not return_uncertainty:
        return p

    if "covariance" in grp:
        uncertainty = grp["covariance"][:]
    elif "uncertainty" in dset.attrs:
        uncertainty = dset.attrs["uncertainty"]
    else:
        uncertainty = None
    return p, uncertainty


def get_keypoints_from_h5(hfile, name: str, return_uncertainty: bool = False):
    return _read_keypoints_from_group(hfile[name], return_uncertainty=return_uncertainty)


class H5KeypointReader:
    """Scoped, read-only keypoint reader with one H5 handle and a bounded LRU cache."""

    def __init__(self, path: Path, *, max_size: int = 512):
        if max_size < 1:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.path = Path(path)
        self.max_size = max_size
        self._hfile = None
        self._cache = OrderedDict()

    def __enter__(self):
        self._hfile = h5py.File(str(self.path), "r", libver="latest")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._cache.clear()
            if self._hfile is not None:
                self._hfile.close()
        finally:
            self._hfile = None
            if self.path.exists():
                drop_page_cache(self.path)
        return False

    def get(self, name: str):
        if self._hfile is None:
            raise RuntimeError("H5KeypointReader must be used as a context manager")
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]

        value = get_keypoints_from_h5(self._hfile, name)
        value.setflags(write=False)
        self._cache[name] = value
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return value


def get_keypoints(path: Path, name: str, return_uncertainty: bool = False) -> np.ndarray:
    """Get keypoints from H5 file.

    Args:
        path: Path to H5 file
        name: Image name (group name in H5)
        return_uncertainty: If True, also return uncertainty/covariance

    Returns:
        If return_uncertainty=False: (N, 2) keypoints array
        If return_uncertainty=True: tuple of (keypoints, uncertainty) where uncertainty is:
            - (N, 3) array [var_x, var_y, cov_xy] if per-keypoint covariance was stored
            - scalar if only scalar uncertainty was stored
            - None if no uncertainty was stored
    """
    with h5py.File(str(path), "r", libver="latest") as hfile:
        return get_keypoints_from_h5(hfile, name, return_uncertainty=return_uncertainty)


def find_pair(hfile, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(f"Could not find pair {(name0, name1)}... " "Maybe you matched with a different list of pairs? ")


def get_matches_from_h5(hfile, name0: str, name1: str) -> tuple[np.ndarray, np.ndarray]:
    pair, reverse = find_pair(hfile, name0, name1)
    matches = hfile[pair]["matches0"].__array__()
    scores = hfile[pair]["matching_scores0"].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    return matches, scores


def get_matches(path: Path, name0: str, name1: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        return get_matches_from_h5(hfile, name0, name1)


def ordered_pair_images(pairs) -> list[str]:
    return list(dict.fromkeys(image for pair in pairs for image in pair))

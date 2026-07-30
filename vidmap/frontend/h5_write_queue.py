"""Asynchronous writers for frontend artifacts."""

import logging
from queue import Queue
from threading import Thread

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class H5WriteQueue:
    """Bounded worker queue that reports write failures at the owner boundary."""

    def __init__(self, write_fn):
        self.queue = Queue(1)
        self.thread = Thread(target=self._write_until_closed, args=(write_fn,))
        self.shutdown = False
        self.errors = []
        self.thread.start()

    def close(self):
        if self.shutdown:
            return
        self.shutdown = True
        self.queue.put(None)
        self.thread.join()
        if self.errors:
            raise RuntimeError(f"H5WriteQueue failed with {len(self.errors)} worker error(s)") from self.errors[0]

    join = close

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as writer_error:
            if exc is None:
                raise
            writer_errors = getattr(exc, "writer_queue_errors", [])
            writer_errors.append(writer_error)
            exc.writer_queue_errors = writer_errors
        return False

    def _write_until_closed(self, write_fn):
        try:
            item = self.queue.get()
            while item is not None:
                try:
                    write_fn(item)
                except Exception as error:
                    self.errors.append(error)
                    logger.error("H5 worker failed: %s", error)
                item = self.queue.get()
        except Exception as error:
            self.errors.append(error)
            logger.error("H5 writer failed: %s", error)

    def put(self, data):
        if not self.shutdown:
            self.queue.put(data)


def write_pair_matches(inp, match_path):
    """Write one sparse-match payload; suitable as a queue entrypoint."""
    pair, pred = inp
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        group = fd.create_group(pair)
        matches = pred["matches0"][0].cpu().short().numpy()
        group.create_dataset("matches0", data=matches)
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().half().numpy()
            group.create_dataset("matching_scores0", data=scores)


def save_features(pred, path, name, uncertainty=None):
    """Write one local-feature group to its H5 artifact."""
    with h5py.File(path, "a", libver="latest") as fd:
        try:
            if name in fd:
                del fd[name]
            group = fd.create_group(name)
            for key, value in sorted(pred.items()):
                group.create_dataset(key, data=value)
            if "keypoints" in pred:
                if uncertainty is None:
                    group["keypoints"].attrs["uncertainty"] = 1.0
                elif np.ndim(uncertainty) == 0 or isinstance(uncertainty, (int, float)):
                    group["keypoints"].attrs["uncertainty"] = float(uncertainty)
                else:
                    group.create_dataset("covariance", data=uncertainty.astype(np.float32))
        except OSError as error:
            if "No space left on device" in error.args[0]:
                logger.error("Out of disk space while storing frontend features")
                del group, fd[name]
            raise


def save_keypoints(pred, sparse_features_path):
    """Write one propagation keypoint payload to the sparse-feature artifact."""
    name = pred["name"][0]
    saved = {key: value for key, value in pred.items() if key in ["keypoints", "scores", "image_size"]}
    saved["keypoints"] = saved["keypoints"][0]
    save_features(saved, str(sparse_features_path), name, pred.get("uncertainty"))

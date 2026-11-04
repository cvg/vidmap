"""Model ownership and identities shared by RoMa and FlexGlue frontends."""

import gc
import logging
from dataclasses import dataclass
from typing import Callable

import torch

from vidmap.utils.device import empty_device_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyframeMatch:
    predict: Callable
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    tracking_match: object = None

    def propagate(self, points, certainty_threshold):
        return self.predict(points, certainty_threshold)


def tracker_type(conf):
    from vidmap.frontend.models.flexglue import FlexGlueModel
    from vidmap.frontend.models.romav2 import RoMaV2Model

    return {"romav2": RoMaV2Model, "flexglue": FlexGlueModel}[conf.name]


def tracker_cache_identity(conf):
    return tracker_type(conf).cache_identity(conf)


def release_tracker(tracker_model, *, suppress_errors):
    errors = []
    for action in (tracker_model.cpu, empty_device_cache, gc.collect):
        try:
            action()
        except BaseException as error:
            errors.append(error)

    if errors:
        logger.warning("Tracker cleanup failed: %s", errors[0])
    else:
        logger.debug("Released owned tracker model")
    if errors and not suppress_errors:
        raise errors[0]


class LazyTracker:
    """Own at most one lazily loaded matching model for one frontend run."""

    def __init__(self, tracker_conf, device: torch.device):
        self._tracker_conf = tracker_conf
        self._device = device
        self._model = None
        self._closed = False

    @property
    def device(self) -> torch.device:
        return self._device

    def parameters(self):
        return self.get().parameters()

    def get(self):
        if self._closed:
            raise RuntimeError("Tracker owner is closed")
        if self._model is None:
            self._model = tracker_type(self._tracker_conf)(self._tracker_conf, self._device).eval().to(self._device)
        return self._model

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Tracker owner is closed")
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        self._closed = True
        if self._model is None:
            return False
        model = self._model
        self._model = None
        release_tracker(model, suppress_errors=exc_type is not None)
        return False


def create_lazy_tracker(tracker_conf, device: torch.device):
    """Return the single lazy tracker owner for one frontend run."""
    return LazyTracker(tracker_conf, device=device)

"""RoMaV2 model owner for streaming frontend."""

import gc
import logging
from dataclasses import dataclass as result_dataclass
from functools import lru_cache
from pathlib import Path

import torch

from vidmap.frontend.cache import file_fingerprint
from vidmap.frontend.models.romav2_inference import (
    match_lowres_batch,
    match_true_highres_pair,
    use_native_local_correlation,
)
from vidmap.frontend.options.matching import RoMaV2Options
from vidmap.model_sources import import_model_package, model_package_root

ROMAV2_SOURCE_REVISION = "95c9968145c8906b7b59383258e9f73b02853d89"
ROMAV2_PACKAGE_ROOT = model_package_root(
    "romav2",
    "third_party/RoMaV2/src/romav2",
)
ROMAV2_SOURCE = ROMAV2_PACKAGE_ROOT.parent
ROMAV2_CHECKPOINT_SHA256 = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"
ROMAV2_VERSION = "v2.0.1"

logger = logging.getLogger(__name__)


def _configure_romav2_logging() -> None:
    """Map RoMaV2's package logger onto VidMap's runtime output policy."""
    dependency_level = logging.DEBUG if logging.getLogger("vidmap").isEnabledFor(logging.DEBUG) else logging.WARNING
    logging.getLogger("romav2").setLevel(dependency_level)


@lru_cache(maxsize=1)
def _verify_romav2_checkpoint() -> None:
    path = Path(torch.hub.get_dir()) / "checkpoints/romav2.0.1.pt"
    if not path.is_file():
        raise RuntimeError(f"RoMaV2 v2.0.1 checkpoint is unavailable: {path}")
    actual = file_fingerprint(path)
    if actual != ROMAV2_CHECKPOINT_SHA256:
        raise RuntimeError(f"RoMaV2 checkpoint has sha256 {actual}, expected {ROMAV2_CHECKPOINT_SHA256}")


@result_dataclass(frozen=True)
class RoMaMatch:
    """Dense RoMa output expressed in target-image pixels."""

    matches: torch.Tensor
    certainty: torch.Tensor
    covariance: torch.Tensor | None = None


@result_dataclass(frozen=True)
class _ParsedRoMaOutput:
    target_warp: torch.Tensor
    certainty: torch.Tensor
    covariance: torch.Tensor | None


class RoMaV2Model(torch.nn.Module):
    """Own the RoMaV2 model used directly by streaming stages."""

    def __init__(self, conf: RoMaV2Options):
        super().__init__()
        assert isinstance(conf, RoMaV2Options), f"Expected RoMaV2Options, got {type(conf).__name__}"
        self.conf = conf
        module = import_model_package("romav2", ROMAV2_PACKAGE_ROOT)
        _configure_romav2_logging()

        cfg = module.RoMaV2.Cfg(
            setting="precise",
            compile=conf.compile,
        )
        use_native_local_correlation()
        # RoMaV2 downloads its release checkpoint on first construction.
        self._net = module.RoMaV2(cfg)
        _verify_romav2_checkpoint()
        self._net.bidirectional = False
        self._net.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, data):
        raise NotImplementedError("Use the explicit RoMaV2 inference operations")

    @torch.no_grad()
    def match_lowres_batch(self, image_a, image_b, *, names_a, names_b, output_size) -> RoMaMatch:
        """Match an ordered low-resolution batch for keyframe selection."""
        assert image_a.shape[0] == image_b.shape[0] == len(names_a) == len(names_b)
        raw = match_lowres_batch(self._net, image_a, image_b)
        return self._pixel_match(raw, output_size=output_size, return_covariance=False)

    @torch.no_grad()
    def match_highres_pair(
        self,
        image_a_lowres,
        image_b_lowres,
        image_a_highres,
        image_b_highres,
        *,
        lowres_resolution,
        return_covariance=False,
    ) -> RoMaMatch:
        """Match one high-resolution pair for tracking or loop closure."""
        highres_height, highres_width = image_a_highres.shape[-2:]
        assert image_a_lowres.shape[-2:] == image_b_lowres.shape[-2:] == (lowres_resolution, lowres_resolution)
        assert image_b_highres.shape[-2:] == (highres_height, highres_width)
        raw = match_true_highres_pair(
            self._net,
            image_a_lowres,
            image_b_lowres,
            image_a_highres,
            image_b_highres,
        )
        return self._pixel_match(
            raw,
            output_size=(highres_width, highres_height),
            return_covariance=return_covariance,
        )

    def _pixel_match(self, raw, *, output_size, return_covariance) -> RoMaMatch:
        parsed = self._parse_output(raw, return_covariance=return_covariance)
        from romav2.geometry import to_pixel

        width, height = output_size
        matches = to_pixel(parsed.target_warp, H=height, W=width)
        return RoMaMatch(matches, parsed.certainty, parsed.covariance)

    def _parse_output(self, raw, *, return_covariance) -> _ParsedRoMaOutput:
        if not isinstance(raw, dict):
            raise TypeError(f"RoMaV2 match output must be a dictionary, got {type(raw).__name__}")
        warp_ab = raw["warp_AB"]
        certainty = raw["overlap_AB"][..., 0]
        covariance = None
        if return_covariance:
            from vidmap.utils.small_matrix import fast_inverse_2x2

            covariance = fast_inverse_2x2(raw["precision_AB"])
        return _ParsedRoMaOutput(warp_ab, certainty, covariance)


def load_romav2_model(conf: RoMaV2Options) -> RoMaV2Model:
    """Construct the sole supported frontend tracker from its typed config."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoMaV2Model(conf).eval().to(device)
    logger.info("Loaded RoMaV2 model")
    return model


def _release_romav2_model(tracker_model, *, suppress_errors):
    def clear_cuda_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    errors = []
    for action in (tracker_model.cpu, clear_cuda_cache, gc.collect):
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


class LazyRoMaV2Tracker:
    """Own at most one lazily loaded RoMaV2 model for one frontend run."""

    def __init__(self, tracker_conf):
        self._tracker_conf = tracker_conf
        self._model = None
        self._closed = False

    def get(self):
        if self._closed:
            raise RuntimeError("RoMaV2 owner is closed")
        if self._model is None:
            self._model = load_romav2_model(self._tracker_conf)
        return self._model

    def __enter__(self):
        if self._closed:
            raise RuntimeError("RoMaV2 owner is closed")
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        self._closed = True
        if self._model is None:
            return False
        model = self._model
        self._model = None
        _release_romav2_model(model, suppress_errors=exc_type is not None)
        return False


def create_lazy_romav2_tracker(tracker_conf):
    """Return the single lazy tracker owner for one frontend run."""
    return LazyRoMaV2Tracker(tracker_conf)


def romav2_cache_identity(conf: RoMaV2Options):
    """Return the semantic configuration and immutable assets used by RoMaV2."""
    assert isinstance(conf, RoMaV2Options), f"Expected RoMaV2Options, got {type(conf).__name__}"
    return {
        "config": {
            "setting": "precise",
            "compile": conf.compile,
            "bidirectional": False,
        },
        "version": ROMAV2_VERSION,
        "source_revision": ROMAV2_SOURCE_REVISION,
        "checkpoint_sha256": ROMAV2_CHECKPOINT_SHA256,
        "true_highres": True,
        "native_local_correlation": True,
    }

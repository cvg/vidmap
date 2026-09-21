"""RoMaV2 model owner for streaming frontend."""

import gc
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from vidmap.frontend.cache import file_fingerprint
from vidmap.frontend.models.romav2_compile_cache import CachedRoMaGraph
from vidmap.frontend.models.romav2_features import (
    FEATURE_NAMES,
    RoMaFeatureMatcher,
    RoMaFeatureProjection,
    RoMaImageFeatures,
)
from vidmap.frontend.options.matching import RoMaV2Options
from vidmap.model_sources import import_model_package, model_package_root

ROMAV2_SOURCE_REVISION = "f23bab45a53ffb3f3c3cdda0566d0de5365e7339"
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


@dataclass(frozen=True)
class RoMaMatch:
    """Dense RoMa output expressed in target-image pixels."""

    matches: torch.Tensor
    certainty: torch.Tensor
    covariance: torch.Tensor | None = None


class RoMaV2Model(torch.nn.Module):
    """Own the RoMaV2 model used directly by streaming stages."""

    def __init__(self, conf: RoMaV2Options):
        super().__init__()
        assert isinstance(conf, RoMaV2Options), f"Expected RoMaV2Options, got {type(conf).__name__}"
        self.conf = conf
        # Register the operator before loading an executable graph in a fresh process.
        from vidmap.frontend.models.romav2_correlation import local_correlation

        module = import_model_package("romav2", ROMAV2_PACKAGE_ROOT)
        _configure_romav2_logging()

        self._source_model = None
        self._feature_buffer = {}
        self._carried_features = None
        self._carried_name = None
        compiled_cuda = conf.compile and torch.cuda.is_available()

        def create_net():
            if self._source_model is not None:
                return self._source_model
            # RoMaV2 downloads its release checkpoint on first construction.
            net = module.RoMaV2(module.RoMaV2.Cfg(setting="precise", compile=False))
            _verify_romav2_checkpoint()
            for refiner in net.refiners.values():
                assert refiner.cfg.grid_sample_mode == "bilinear"
                refiner.correlation = local_correlation
            self._source_model = net.eval().requires_grad_(False)
            return self._source_model

        for name, component, adapter in (
            ("_feature_extractor", "image_features", RoMaImageFeatures),
            ("_feature_matcher", "feature_matcher", RoMaFeatureMatcher),
            ("_feature_projector", "feature_projection", RoMaFeatureProjection),
        ):
            if compiled_cuda:
                graph = CachedRoMaGraph(
                    lambda adapter=adapter: adapter(create_net()).eval(),
                    romav2_cache_identity(conf),
                    ROMAV2_PACKAGE_ROOT,
                    component=component,
                ).eval()
            else:
                graph = adapter(create_net()).eval()
                if conf.compile:
                    graph = torch.compile(graph)
            setattr(self, name, graph)

    def _apply(self, fn, recurse=True):
        self._feature_buffer.clear()
        self._carried_features = None
        self._carried_name = None
        self._source_model = None
        return super()._apply(fn, recurse=recurse)

    def _extract_features(self, low, high=None):
        images = (low,) if high is None else (low, high)
        # Own contiguous inference tensors, even when inputs are already on the GPU.
        result = self._feature_extractor(
            *(image.cuda(non_blocking=True).clone(memory_format=torch.contiguous_format) for image in images)
        )
        names = FEATURE_NAMES[:5] if high is None else FEATURE_NAMES
        projected = self._feature_projector(*(result[name] for name in names[2:]))
        return (*(result[name] for name in names[:2]), *(projected[name] for name in names[2:]))

    def forward(self, data):
        raise NotImplementedError("Use the explicit RoMaV2 inference operations")

    @torch.inference_mode()
    def match_lowres_batch(self, image_a, image_b, *, names_a, names_b, output_size, batch_size) -> RoMaMatch:
        """Pad to a fixed inference batch, returning only the real ordered pairs."""
        assert image_a.shape[0] == image_b.shape[0] == len(names_a) == len(names_b)
        count = image_a.shape[0]
        if not 0 < count <= batch_size:
            raise ValueError("Low-resolution pair count must be between one and batch_size")
        if count < batch_size:
            # Keep the final batch on the same compiled graph as full batches.
            image_b = torch.cat((image_b, image_b.new_zeros((batch_size - count, *image_b.shape[1:]))))
        assert list(names_a[1:]) == list(names_b[:-1]), "Feature reuse requires consecutive frame pairs"
        if self._carried_features is None:
            seed = torch.cat((image_a[:1], image_a.new_zeros((batch_size - 1, *image_a.shape[1:]))))
            initial = self._extract_features(seed)
            self._carried_features = tuple(value[:1].clone() for value in initial)
            self._carried_name = names_a[0]
        assert self._carried_name == names_a[0], "Keyframe batches must preserve frame continuity"
        features_b = self._extract_features(image_b)
        # A uses the preceding frame's features; keep padded rows in place.
        features_a = tuple(
            torch.cat((previous, new[: count - 1], new[count:]))
            for previous, new in zip(self._carried_features, features_b)
        )
        raw = self._feature_matcher(*features_a, *features_b)
        self._carried_features = tuple(value[count - 1 : count].clone() for value in features_b)
        self._carried_name = names_b[-1]
        output = self._pixel_match(raw, output_size=output_size, return_covariance=False)
        return RoMaMatch(output.matches[:count], output.certainty[:count])

    @torch.inference_mode()
    def match_highres_pair(
        self,
        image_a_lowres,
        image_b_lowres,
        image_a_highres,
        image_b_highres,
        *,
        names,
        retained_names,
        lowres_resolution,
        return_covariance=False,
    ) -> RoMaMatch:
        """Reuse features only while a view belongs to the caller's live window."""
        live = set(retained_names)
        assert len(names) == 2 and set(names) <= live
        assert image_a_lowres.shape[-2:] == image_b_lowres.shape[-2:] == (lowres_resolution, lowres_resolution)
        assert image_a_highres.shape == image_b_highres.shape and image_a_highres.shape[0] == 1
        # Release features carried over from keyframe selection.
        self._carried_features = None
        self._carried_name = None
        for name in self._feature_buffer.keys() - live:
            del self._feature_buffer[name]
        for name, low, high in zip(names, (image_a_lowres, image_b_lowres), (image_a_highres, image_b_highres)):
            if name not in self._feature_buffer:
                self._feature_buffer[name] = self._extract_features(low, high)
        features_a, features_b = (self._feature_buffer[name] for name in names)
        raw = self._feature_matcher(*features_a, *features_b)
        return self._pixel_match(
            raw,
            output_size=tuple(image_a_highres.shape[-2:][::-1]),
            return_covariance=return_covariance,
        )

    def _pixel_match(self, raw, *, output_size, return_covariance) -> RoMaMatch:
        from romav2.geometry import to_pixel
        from romav2.romav2 import _map_confidence

        if not isinstance(raw, dict):
            raise TypeError(f"RoMaV2 forward output must be a dictionary, got {type(raw).__name__}")
        overlap, precision = _map_confidence(confidence=raw["confidence_AB"], threshold=None)
        covariance = None
        if return_covariance:
            from vidmap.utils.small_matrix import fast_inverse_2x2

            covariance = fast_inverse_2x2(precision)
        width, height = output_size
        matches = to_pixel(raw["warp_AB"], H=height, W=width)
        return RoMaMatch(matches, overlap[..., 0], covariance)


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
        },
        "version": ROMAV2_VERSION,
        "torch": torch.__version__,
        "source_revision": ROMAV2_SOURCE_REVISION,
        "checkpoint_sha256": ROMAV2_CHECKPOINT_SHA256,
        "true_highres": True,
    }

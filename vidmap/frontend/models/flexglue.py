"""FlexGlue keyframe queries and RoMa-refined tracking in working-image pixels."""

from contextlib import contextmanager
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from vidmap.frontend.cache import file_fingerprint
from vidmap.frontend.models.tracker import KeyframeMatch
from vidmap.frontend.options.matching import FlexGlueOptions

FLEXGLUE_WEIGHTS_URL = "https://github.com/cvg/FlexGlue/releases/download/v0.1/flexglue.pth"
FLEXGLUE_CHECKPOINT_SHA256 = "d7371f743b8cdc4adf723c2aa05b181587fe4bc177fced139697975dff790068"


def flexglue_cache_identity(conf, *, keyframing=False):
    from importlib.util import find_spec

    from vidmap.frontend.models.romav2 import ROMAV2_CHECKPOINT_SHA256, ROMAV2_SOURCE_REVISION

    spec = find_spec("flexglue")
    if spec is None:
        raise ModuleNotFoundError("Install FlexGlue separately before selecting the FlexGlue frontend")
    source = Path(spec.origin).parent
    config = asdict(conf)
    if keyframing:
        config = {
            "name": conf.name,
            "keyframing_layers": conf.keyframing_layers,
            "compile": conf.compile,
        }

    identity = {
        "config": config,
        "source_sha256": {
            str(path.relative_to(source)): file_fingerprint(path) for path in sorted(source.rglob("*.py"))
        },
        "checkpoint_sha256": FLEXGLUE_CHECKPOINT_SHA256,
        "keyframe_visibility": "finite-in-bounds",
        "torch": torch.__version__,
    }
    if not keyframing:
        identity["refiner"] = (ROMAV2_SOURCE_REVISION, ROMAV2_CHECKPOINT_SHA256)
    return identity


class FlexGlueModel(torch.nn.Module):
    def __init__(self, conf: FlexGlueOptions, device):
        super().__init__()
        from flexglue import FlexGlue

        self.conf = conf
        self.device = device
        weights = Path(torch.hub.get_dir()) / "checkpoints/flexglue.pth"
        if not weights.is_file():
            weights.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(FLEXGLUE_WEIGHTS_URL, str(weights))
        if file_fingerprint(weights) != FLEXGLUE_CHECKPOINT_SHA256:
            raise ValueError("Unexpected FlexGlue release checkpoint")
        self.net = (
            FlexGlue(
                weights=str(weights.resolve()),
                n_layers=max(conf.keyframing_layers, conf.tracking_layers, conf.retrieval_layers),
                refine=False,
            )
            .eval()
            .to(device)
        )
        if conf.compile:
            self.net.compile()
        self.features = {}
        from vidmap.frontend.models.romav2_refinement import RoMaRefinement

        self.refinement = RoMaRefinement(device, compile=conf.compile).eval().to(device)

    needs_lowres_images = False
    normalize_keyframe_images = False

    @staticmethod
    def cache_identity(conf):
        return flexglue_cache_identity(conf)

    @staticmethod
    def keyframe_cache_identity(conf, track_options):
        return flexglue_cache_identity(conf, keyframing=True)

    @staticmethod
    def keyframe_image_options(options):
        if options.resize_to_shape is None:
            options = replace(options, resize_to_shape=(512, 512))
        if options.resolution % 32:
            raise ValueError("FlexGlue low-resolution input size must be a multiple of 32")
        return options

    def prefetch_images(self, inputs, image_keys, *, image_names=None):
        def needs_transfer(item, key):
            name = item[image_names[key]]
            return (name, tuple(item[key].shape[-2:])) not in self.features or name not in self.refinement.features

        yield from _prefetch_images(
            inputs, image_keys, self.device, needs_transfer=None if image_names is None else needs_transfer
        )

    @contextmanager
    def keyframe_matches(self, loader, original_width, original_height, *, batch_size, coarse_only=False):
        try:
            yield (
                self.keyframe_pairs(batch, (original_width, original_height), batch_size=batch_size)
                for batch in loader
            )
        finally:
            self.features.clear()

    def keyframe_match(self, context, source_size, target_size):
        return KeyframeMatch(
            partial(self._predict_keypoints, context, source_size, target_size), source_size, target_size
        )

    @torch.inference_mode()
    def _predict_keypoints(self, context, source_size, target_size, keypoints, certainty_threshold):
        from flexglue.utils import ops

        points = keypoints.detach().cpu().numpy() if torch.is_tensor(keypoints) else np.asarray(keypoints)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if not len(points):
            return torch.from_numpy(points), torch.empty(0), torch.empty(0, dtype=torch.bool)
        device = context.cf[0].device
        queries = torch.as_tensor(points, device=device)[None]
        predicted, confidence = self.net.run_head(
            context.layer,
            ops.normalize_keypoints(queries + 0.5, torch.tensor([source_size], device=device)),
            *context.direction(0),
        )
        predicted = (
            ops.denormalize_keypoints(predicted, torch.tensor([target_size], device=device))[0].cpu().numpy() - 0.5
        )
        confidence = confidence[0].cpu().numpy()
        valid = np.isfinite(predicted).all(1) & np.isfinite(confidence)
        valid &= ((predicted >= -0.5) & (predicted < np.asarray(target_size) - 0.5)).all(1)
        confidence = np.where(valid, confidence, 0).astype(np.float32)
        return (
            torch.from_numpy(predicted),
            torch.from_numpy(confidence),
            # Preserve benchmarked keyframing without confidence filtering.
            torch.from_numpy(valid),
        )

    def _apply(self, fn, recurse=True):
        self.features.clear()
        return super()._apply(fn, recurse=recurse)

    @torch.inference_mode()
    def match_highres_pair(
        self,
        low0,
        low1,
        high0,
        high1,
        *,
        names,
        retained_names,
        lowres_resolution,
        return_covariance=False,
        protect=False,
        keyframing=False,
        retrieval=False,
    ):
        source_size = tuple(high0.shape[-2:][::-1])
        target_size = tuple(high1.shape[-2:][::-1])
        layers = (
            self.conf.keyframing_layers
            if keyframing
            else self.conf.retrieval_layers if retrieval else self.conf.tracking_layers
        )
        self.features = {key: value for key, value in self.features.items() if key[0] in retained_names}
        data = []
        for name, image, size in zip(names, (high0, high1), (source_size, target_size)):
            key = (name, tuple(image.shape[-2:]))
            if key not in self.features:
                self.features[key] = self.net.descriptor(
                    {"image": image.to(self.device)}, resize=(lowres_resolution,) * 2
                )
            data.append({**self.features[key], "image_size": torch.tensor([size], device=self.device)})
        reject = not (protect or keyframing) and self.conf.pair_policy == "reject"
        config = self.net.resolve_conf(
            {
                "n_layers": layers,
                "early_stop_decay": self.conf.early_stop_decay if reject else None,
                "min_covisibility": self.conf.pair_threshold if reject else 0.0,
            }
        )
        # Upstream decay uses the stack length; restrict it to this stage's depth.
        transformers = self.net.transformers
        self.net.transformers = transformers[:layers]
        try:
            context = self.net.decode_pair(*data, config)
        finally:
            self.net.transformers = transformers
        score = float(context.covisibility_score[0]) if reject else 1.0
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid FlexGlue pair probability")
        if reject and score < self.conf.pair_threshold:
            return None
        if keyframing:
            return self.keyframe_match(context, source_size, target_size)
        coarse = self.net.grid_correspondences(context, 0, *data, config)
        return self.refinement(
            coarse["dense_p2di_nj"],
            coarse["dense_confidencei"],
            high0,
            high1,
            names=names,
            retained_names=retained_names,
        )

    @torch.inference_mode()
    def keyframe_pairs(self, batch, output_size, *, batch_size):
        names_a, names_b = batch["names_A"], batch["names_B"]
        images_a, images_b = batch["im_A_batch"], batch["im_B_batch"]
        count = len(names_a)
        assert images_a.shape == images_b.shape and count == len(images_a) == len(names_b)
        assert 0 < count <= batch_size
        assert names_a[1:] == names_b[:-1], "Feature reuse requires consecutive frame pairs"
        shape = tuple(images_a.shape[-2:])
        key = (names_a[0], shape)
        if key not in self.features:
            self.features[key] = self.net.descriptor({"image": images_a[:1].to(self.device)}, resize=shape)
        if count < batch_size:
            images_b = torch.cat((images_b, images_b.new_zeros((batch_size - count, *images_b.shape[1:]))))
        following = self.net.descriptor({"image": images_b.to(self.device)}, resize=shape)["features"]
        previous = torch.cat((self.features[key]["features"], following[: count - 1], following[count:]))
        size = following.new_tensor(output_size).expand(batch_size, -1)
        config = self.net.resolve_conf(
            {
                "n_layers": self.conf.keyframing_layers,
                "early_stop_decay": None,
                "min_covisibility": 0.0,
            }
        )
        context = self.net.decode_pair(
            {"features": previous, "image_size": size}, {"features": following, "image_size": size}, config
        )
        self.features = {(names_b[-1], shape): {"features": following[count - 1 : count].clone()}}
        for i in range(count):
            single = context._replace(
                covisibility_score=context.covisibility_score[i : i + 1],
                cf=tuple(value[i : i + 1] for value in context.cf),
            )
            yield self.keyframe_match(single, output_size, output_size)


def _prefetch_images(inputs, image_keys, device, *, needs_transfer=None):
    if device.type != "cuda":
        yield from inputs
        return
    source = iter(inputs)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))

    def launch(item):
        with torch.cuda.stream(stream):
            result = dict(item)
            for key in image_keys:
                if item[key] is not None and (needs_transfer is None or needs_transfer(item, key)):
                    result[key] = item[key].to(device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
        return result, ready

    try:
        first = next(source, None)
        pending = None if first is None else launch(first)
        while pending is not None:
            item, ready = pending
            consumer = torch.cuda.current_stream(device)
            consumer.wait_event(ready)
            for key in image_keys:
                if item[key] is not None and item[key].is_cuda:
                    item[key].record_stream(consumer)
            following = next(source, None)
            pending = None if following is None else launch(following)
            yield item
    finally:
        stream.synchronize()

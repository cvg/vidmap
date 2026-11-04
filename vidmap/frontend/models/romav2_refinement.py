"""Refine a supplied coarse field with pinned RoMaV2 layers and per-view caching."""

from pathlib import Path

import torch

from vidmap.frontend.cache import file_fingerprint
from vidmap.frontend.models.romav2 import ROMAV2_CHECKPOINT_SHA256, ROMAV2_PACKAGE_ROOT, RoMaMatch
from vidmap.frontend.models.romav2_features import RoMaRefinementHead
from vidmap.model_sources import import_model_package


class RoMaRefinement(torch.nn.Module):
    def __init__(self, device, *, compile=False):
        super().__init__()
        import_model_package("romav2", ROMAV2_PACKAGE_ROOT)
        from romav2.features import FineFeatures
        from romav2.refiner import Refiners

        self.device = device
        self.extractor = FineFeatures(FineFeatures.Cfg())
        self.head = RoMaRefinementHead(Refiners(Refiners.Cfg()))
        weights = torch.hub.load_state_dict_from_url(
            "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt", map_location="cpu"
        )
        path = Path(torch.hub.get_dir()) / "checkpoints/romav2.0.1.pt"
        if file_fingerprint(path) != ROMAV2_CHECKPOINT_SHA256:
            raise ValueError("Unexpected RoMa refinement checkpoint")
        self.extractor.load_state_dict(
            {k.removeprefix("refiner_features."): v for k, v in weights.items() if k.startswith("refiner_features.")}
        )
        self.head.refiners.load_state_dict(
            {k.removeprefix("refiners."): v for k, v in weights.items() if k.startswith("refiners.")}
        )
        if device.type == "cuda":
            from vidmap.frontend.models.romav2_correlation import local_correlation

            for refiner in self.head.refiners.values():
                refiner.correlation = local_correlation
        self.features = {}
        if compile and device.type == "cuda":
            self.extractor.forward = torch.compile(self.extractor.forward, fullgraph=True, dynamic=False)
            self._refine = torch.compile(self._refine, fullgraph=True, dynamic=False)

    def _apply(self, fn, recurse=True):
        self.features.clear()
        return super()._apply(fn, recurse=recurse)

    @torch.inference_mode()
    def forward(self, warp, confidence, image0, image1, *, names, retained_names):
        from romav2.geometry import prec_mat_from_prec_params, to_pixel

        if image0.shape != image1.shape:
            raise ValueError("RoMa refinement requires equal working image sizes")
        self.features = {name: value for name, value in self.features.items() if name in retained_names}
        for name, image in zip(names, (image0, image1)):
            if name not in self.features:
                raw = self.extractor(image.to(self.device))
                self.features[name] = {
                    int(s): refiner.proj(raw[int(s)].float()).contiguous() for s, refiner in self.head.refiners.items()
                }
        f0, f1 = (self.features[name] for name in names)
        _, height, width, _ = f0[1].shape
        warp, confidence = self._refine(f0, f1, warp, confidence)
        precision = prec_mat_from_prec_params(confidence[..., 1:4])
        from vidmap.utils.small_matrix import fast_inverse_2x2

        return RoMaMatch(
            to_pixel(warp, H=height, W=width),
            confidence[..., 0].sigmoid(),
            fast_inverse_2x2(precision),
        )

    def _refine(self, f0, f1, warp, confidence):
        return self.head(f0, f1, warp, torch.logit(confidence, eps=1e-6)[..., None])

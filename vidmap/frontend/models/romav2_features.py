"""Per-view features and pair-dependent inference for the pinned RoMaV2 model."""

import torch

FEATURE_NAMES = ("descriptor_11", "descriptor_17", "low_1", "low_2", "low_4", "high_1", "high_2", "high_4")


class RoMaImageFeatures(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.descriptor = net.f
        self.fine = net.refiner_features

    def forward(self, low, high=None, *, coarse_only=False):
        assert isinstance(coarse_only, bool)
        assert not coarse_only or high is None
        output = tuple(self.descriptor(low))
        if not coarse_only:
            fine = self.fine(low)
            output = (*output, *(fine[k] for k in (1, 2, 4)))
            if high is not None:
                fine_high = self.fine(high)
                output = (*output, *(fine_high[k] for k in (1, 2, 4)))
        return {name: value.contiguous() for name, value in zip(FEATURE_NAMES, output)}


class RoMaFeatureProjection(torch.nn.Module):
    """Project once per view, preserving the extractor's BF16 output boundary.

    Keep this separate from the extractor graph: compiling them together lets
    Inductor remove intermediate rounding and changes the refinement inputs.
    Only the projected representation remains in the live feature buffer.
    """

    def __init__(self, net):
        super().__init__()
        self.projections = torch.nn.ModuleDict({key: refiner.proj for key, refiner in net.refiners.items()})

    def forward(self, *features):
        assert len(features) in (3, 6)
        output = []
        for index, value in enumerate(features):
            batch, height, width, channels = value.shape
            projection = self.projections[str((1, 2, 4)[index % 3])]
            output.append(
                projection(value.reshape(batch, height * width, channels).float())
                .reshape(batch, height, width, -1)
                .contiguous()
            )
        return dict(zip(FEATURE_NAMES[2:], output))


class RoMaRefinementHead(torch.nn.Module):
    """Refine coarse predictions using projected per-image features."""

    def __init__(self, refiners, anchor_width=512, anchor_height=512):
        super().__init__()
        self.refiners = refiners
        self.anchor_width = anchor_width
        self.anchor_height = anchor_height

    def forward(self, features_a, features_b, warp, confidence, *, reset_precision=False):
        from romav2.romav2 import _interpolate_warp_and_confidence

        _, height, width, _ = features_a[1].shape
        scale = torch.tensor((width / self.anchor_width, height / self.anchor_height), device=warp.device)
        for stride, refiner in self.refiners.items():
            warp, confidence = _interpolate_warp_and_confidence(
                warp=warp,
                confidence=confidence,
                H=height,
                W=width,
                patch_size=int(stride),
                zero_out_precision=reset_precision and int(stride) == 4,
            )
            prediction = refiner(
                f_A=features_a[int(stride)],
                f_B=features_b[int(stride)],
                prev_warp=warp,
                prev_confidence=confidence,
                scale_factor=scale,
                projected=True,
            )
            warp, confidence = prediction["warp"], prediction["confidence"]
        return warp, confidence


class RoMaFeatureMatcher(torch.nn.Module):
    """Match supplied features, with refinement controlled explicitly by the caller."""

    def __init__(self, net):
        super().__init__()
        self.matcher = net.matcher
        self.refinement = RoMaRefinementHead(net.refiners, net.anchor_width, net.anchor_height)

    def forward(self, features_a, features_b, *, refine=True):
        from romav2.romav2 import _highest_matmul_precision

        assert isinstance(refine, bool)
        assert len(features_a) == len(features_b)
        assert len(features_a) in ((5, 8) if refine else (2, 5, 8))
        prediction = self.matcher(
            list(features_a[:2]), list(features_b[:2]), img_A=None, img_B=None, bidirectional=False
        )
        if not refine:
            return prediction
        if not _highest_matmul_precision():
            raise RuntimeError("Float32 matmul precision must be set to highest")
        assert not self.training, "Currently only inference mode released"
        warp, confidence = prediction["warp_AB"], prediction["confidence_AB"]
        for offset in range(2, len(features_a), 3):
            warp, confidence = self.refinement(
                dict(zip((1, 2, 4), features_a[offset : offset + 3])),
                dict(zip((1, 2, 4), features_b[offset : offset + 3])),
                warp,
                confidence,
                reset_precision=offset == 5,
            )
        return dict(warp_AB=warp, confidence_AB=confidence, warp_BA=None, confidence_BA=None)

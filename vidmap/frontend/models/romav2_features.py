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


class RoMaFeatureMatcher(torch.nn.Module):
    """Match supplied features, with refinement controlled explicitly by the caller."""

    bidirectional = False
    return_intermediates = False

    def __init__(self, net):
        super().__init__()
        self.matcher = net.matcher
        self.refiners = net.refiners
        self.anchor_width = net.anchor_width
        self.anchor_height = net.anchor_height

    def forward(self, features_a, features_b, *, refine=True):
        from romav2.romav2 import RoMaV2

        assert isinstance(refine, bool)
        assert len(features_a) == len(features_b)
        assert len(features_a) in ((5, 8) if refine else (2, 5, 8))
        if not refine:
            return self.matcher(
                list(features_a[:2]), list(features_b[:2]), img_A=None, img_B=None, bidirectional=False
            )
        return RoMaV2.forward(self, None, None, features_A=features_a, features_B=features_b, projected=True)

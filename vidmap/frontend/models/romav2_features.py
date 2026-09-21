"""Per-view features and pair-dependent inference for the pinned RoMaV2 model."""

import torch

FEATURE_NAMES = ("descriptor_11", "descriptor_17", "low_1", "low_2", "low_4", "high_1", "high_2", "high_4")


class RoMaImageFeatures(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.descriptor = net.f
        self.fine = net.refiner_features

    def forward(self, low, high=None):
        descriptor = tuple(self.descriptor(low))
        fine = self.fine(low)
        output = (*descriptor, *(fine[k] for k in (1, 2, 4)))
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
    """Run upstream matching/refinement on already-projected per-view features."""

    bidirectional = False
    return_intermediates = False

    def __init__(self, net):
        super().__init__()
        self.matcher = net.matcher
        self.refiners = net.refiners
        self.anchor_width = net.anchor_width
        self.anchor_height = net.anchor_height

    def forward(self, *features):
        from romav2.romav2 import RoMaV2

        assert len(features) in (10, 16)
        count = len(features) // 2
        return RoMaV2.forward(
            self, None, None, features_A=features[:count], features_B=features[count:], projected=True
        )

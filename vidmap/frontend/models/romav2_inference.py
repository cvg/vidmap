"""VidMap inference helpers for the public RoMaV2 implementation."""

import importlib

import torch
import torch.nn.functional as F


def use_native_local_correlation() -> None:
    """Select RoMaV2's deterministic PyTorch correlation implementation."""
    correlation = importlib.import_module("romav2.local_correlation")
    correlation.local_corr = None


def _resize(image: torch.Tensor, size: tuple[int, int]):
    return F.interpolate(
        image,
        size=size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


def _map_confidence(confidence: torch.Tensor, threshold: float | None):
    from romav2.geometry import prec_mat_from_prec_params

    overlap = confidence[..., :1].sigmoid()
    if threshold is not None:
        overlap[overlap > threshold] = 1.0
    precision = prec_mat_from_prec_params(confidence[..., 1:4])
    return overlap, precision


def _finalize_predictions(model, predictions: dict[str, torch.Tensor]):
    if not isinstance(predictions, dict):
        raise TypeError(f"RoMaV2 forward output must be a dictionary, got {type(predictions).__name__}")
    confidence_ab = predictions["confidence_AB"]
    overlap_ab, precision_ab = _map_confidence(confidence_ab, model.threshold)
    if model.bidirectional:
        confidence_ba = predictions["confidence_BA"]
        overlap_ba, precision_ba = _map_confidence(confidence_ba, model.threshold)
    else:
        confidence_ba = None
        overlap_ba = None
        precision_ba = None

    warp_ab = predictions["warp_AB"]
    warp_ba = predictions["warp_BA"]
    return {
        "warp_AB": warp_ab.clone(),
        "confidence_AB": confidence_ab.clone(),
        "overlap_AB": overlap_ab.clone(),
        "precision_AB": precision_ab.clone(),
        "warp_BA": warp_ba.clone() if warp_ba is not None else None,
        "confidence_BA": confidence_ba.clone() if confidence_ba is not None else None,
        "overlap_BA": overlap_ba.clone() if overlap_ba is not None else None,
        "precision_BA": precision_ba.clone() if precision_ba is not None else None,
    }


@torch.inference_mode()
def match_lowres_batch(model, image_a: torch.Tensor, image_b: torch.Tensor):
    """Run public RoMaV2 on an already resized low-resolution batch."""
    target_size = image_a.shape[-2:]
    predictions = model(_resize(image_a, target_size), _resize(image_b, target_size))
    return _finalize_predictions(model, predictions)


@torch.inference_mode()
def match_true_highres_pair(
    model,
    image_a_lowres: torch.Tensor,
    image_b_lowres: torch.Tensor,
    image_a_highres: torch.Tensor,
    image_b_highres: torch.Tensor,
):
    """Run public RoMaV2 with VidMap's independently loaded high-res pair."""
    lowres_size = image_a_lowres.shape[-2:]
    highres_size = image_a_highres.shape[-2:]
    predictions = model(
        _resize(image_a_lowres, lowres_size),
        _resize(image_b_lowres, lowres_size),
        img_A_hr=_resize(image_a_highres, highres_size),
        img_B_hr=_resize(image_b_highres, highres_size),
    )
    return _finalize_predictions(model, predictions)

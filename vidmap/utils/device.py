"""Centralized PyTorch device resolution, autocast, and memory management."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an execution device identifier to a canonical torch.device.

    Accepts 'auto', 'cpu', 'cuda', 'gpu', 'mps', or torch.device instances.
    When omitted or 'auto', prioritizes CUDA if available, then MPS, then CPU.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    norm = str(device).strip().lower()
    if norm in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({device!r}), but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if norm == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(
                f"Apple MPS device requested ({device!r}), but torch.backends.mps.is_available() is False."
            )
        return torch.device("mps")
    if norm == "cpu":
        return torch.device("cpu")
    return torch.device(norm)


def get_autocast_context(device: torch.device) -> tuple[AbstractContextManager, torch.dtype]:
    """Return the optimal inference autocast context and compute dtype for the given device."""
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype), dtype
    return nullcontext(), torch.float32


def empty_device_cache(device: torch.device | None = None) -> None:
    """Safely release cached device memory for CUDA or MPS."""
    if (device is None or device.type == "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (
        (device is None or device.type == "mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()

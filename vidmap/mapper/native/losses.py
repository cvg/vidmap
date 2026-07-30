"""Shared construction of native robust-loss records."""

from vidmap.mapper.native.extension import native
from vidmap.mapper.options.positioning import LossConfig


def native_loss_type(name: str):
    return {
        "trivial": native.LossFunctionType.TRIVIAL,
        "huber": native.LossFunctionType.HUBER,
        "cauchy": native.LossFunctionType.CAUCHY,
        "soft_l1": native.LossFunctionType.SOFT_L1,
    }[name]


def _build_loss_config(loss_type, *, scale: float, weight: float):
    loss_config = native.LossConfig()
    loss_config.type = loss_type
    loss_config.scale = float(scale)
    loss_config.weight = float(weight)
    return loss_config


def loss_config_from_options(loss: LossConfig, *, weight: float | None = None):
    return _build_loss_config(
        native_loss_type(loss.name),
        scale=loss.scale,
        weight=loss.weight if weight is None else weight,
    )


def build_named_loss_config(name: str, *, scale: float = 1.0, weight: float = 1.0):
    return _build_loss_config(native_loss_type(name), scale=scale, weight=weight)


def build_typed_loss_config(loss_type, *, scale: float, weight: float):
    return _build_loss_config(loss_type, scale=scale, weight=weight)

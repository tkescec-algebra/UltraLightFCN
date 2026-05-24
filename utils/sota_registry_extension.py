"""Additive SMP comparator-extension registry.

The validated original SOTA registry in `utils.sota_registry` remains unchanged;
this module is intended for future extension runners that opt in explicitly.
"""

from __future__ import annotations

from typing import Any, Dict

from utils.sota_registry import SOTA_REGIMES, split_smp_encoder_decoder_params


SOTA_EXTENSION_MODELS: Dict[str, Dict[str, Any]] = {
    "unet_mobilenetv2": dict(
        model_key="unet",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
    ),
    "dlv3p_efficientnetb0": dict(
        model_key="dlv3p",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
    ),
    "unet_efficientnetb0": dict(
        model_key="unet",
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
    ),
}


__all__ = [
    "SOTA_EXTENSION_MODELS",
    "SOTA_REGIMES",
    "split_smp_encoder_decoder_params",
]

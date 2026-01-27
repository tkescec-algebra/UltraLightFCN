# utils/sota_registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch


# --- Fixed SOTA baselines
SOTA_MODELS: Dict[str, Dict[str, Any]] = {
    "dlv3p_resnet50": dict(
        model_key="dlv3p",
        encoder_name="resnet50",
        encoder_weights="imagenet",
    ),
    "dlv3p_mobilenetv2": dict(
        model_key="dlv3p",
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
    ),
    "unet_resnet34": dict(
        model_key="unet",
        encoder_name="resnet34",
        encoder_weights="imagenet",
    ),
}


# --- Two regimes (predefined; no TEST decisions)
# MinFT: minimal encoder adaptation (small LR, not freeze)
# FullFT: same-style fine-tuning as your protocol (encoder LR scaled)
SOTA_REGIMES: Dict[str, Dict[str, Any]] = {
    "minft": dict(enc_lr_mult=0.1, freeze_encoder=False),
    "fullft": dict(enc_lr_mult=None, freeze_encoder=False),  # enc_lr_mult comes from Phase-5 winner
}


def split_smp_encoder_decoder_params(model: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    SMP models have standard modules:
      - model.encoder
      - model.decoder
      - model.segmentation_head
      - optionally model.classification_head (usually None)
    """
    if not hasattr(model, "encoder") or not hasattr(model, "decoder") or not hasattr(model, "segmentation_head"):
        raise RuntimeError("Expected an SMP model with encoder/decoder/segmentation_head attributes.")

    enc_params = list(model.encoder.parameters())

    dec_params = []
    dec_params += list(model.decoder.parameters())
    dec_params += list(model.segmentation_head.parameters())

    if hasattr(model, "classification_head") and model.classification_head is not None:
        dec_params += list(model.classification_head.parameters())

    return enc_params, dec_params


def freeze_smp_encoder_(model: torch.nn.Module) -> None:
    """Freeze SMP encoder (if you ever want the strict-freeze baseline)."""
    for p in model.encoder.parameters():
        p.requires_grad = False

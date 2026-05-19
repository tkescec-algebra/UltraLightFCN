from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import torch

from utils.load_simclr_pretrain_encoder import _extract_state_dict, _strip_known_wrappers
from utils.ultralight_variant_registry import get_ultralight_variant_spec


WRAPPER_PREFIXES: tuple[str, ...] = (
    "module.",
    "model.",
    "net.",
    "network.",
    "encoder.",
    "backbone.",
    "online_network.",
    "student.",
)

BACKBONE_ENCODER_PREFIXES: tuple[str, ...] = (
    "block1",
    "dsconv2",
    "dsconv3",
    "dilconv4",
    "dilconv5",
)


def _get_prefix(key: str) -> str:
    return key.split(".", 1)[0]


def _make_audit(
    ckpt_path: str,
    variant_name: str,
    loaded_keys: list[str],
    skipped_missing_keys: list[str],
    skipped_shape_keys: list[str],
    missing_required_target_keys: list[str] | None = None,
) -> dict[str, Any]:
    loaded_prefixes = sorted({_get_prefix(key) for key in loaded_keys})
    skipped_prefixes = sorted(
        {_get_prefix(key) for key in skipped_missing_keys + skipped_shape_keys}
    )
    return {
        "checkpoint_path": ckpt_path,
        "variant_name": variant_name,
        "loaded_key_count": len(loaded_keys),
        "skipped_missing_key_count": len(skipped_missing_keys),
        "skipped_shape_key_count": len(skipped_shape_keys),
        "loaded_prefixes": loaded_prefixes,
        "skipped_prefixes": skipped_prefixes,
        "skipped_missing_keys": skipped_missing_keys,
        "skipped_shape_keys": skipped_shape_keys,
        "missing_required_target_keys": list(missing_required_target_keys or []),
    }


def load_pretrained_encoder_into_ultralight_experimental(
    model: torch.nn.Module,
    ckpt_path: str,
    variant_name: str,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Load Phase-3 SimCLR encoder weights with a strict audit policy for
    experimental UltraLightFCN variants.
    """

    if (ckpt_path is None) or (not os.path.isfile(ckpt_path)):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    spec = get_ultralight_variant_spec(variant_name)
    compat = spec.simclr_compatibility_policy
    required_prefixes = tuple(compat["required_encoder_prefixes"])
    allowed_missing_prefixes = set(compat["allowed_missing_prefixes"])

    ckpt = torch.load(ckpt_path, map_location="cpu")
    src_sd = _extract_state_dict(ckpt)
    tgt_sd = model.state_dict()

    loaded_sd: dict[str, torch.Tensor] = {}
    loaded_keys: list[str] = []
    skipped_missing_keys: list[str] = []
    skipped_shape_keys: list[str] = []
    loaded_by_prefix: dict[str, int] = defaultdict(int)
    expected_target_encoder_keys = sorted(
        key for key in tgt_sd.keys() if _get_prefix(key) in required_prefixes
    )

    encoder_seen = 0
    for source_key, value in src_sd.items():
        target_key = _strip_known_wrappers(source_key, WRAPPER_PREFIXES)
        prefix = _get_prefix(target_key)

        if prefix not in required_prefixes and prefix not in allowed_missing_prefixes:
            continue

        encoder_seen += 1

        if target_key not in tgt_sd:
            skipped_missing_keys.append(target_key)
            continue

        if tgt_sd[target_key].shape != value.shape:
            skipped_shape_keys.append(target_key)
            continue

        loaded_sd[target_key] = value
        loaded_keys.append(target_key)
        loaded_by_prefix[prefix] += 1

    if skipped_shape_keys:
        audit = _make_audit(
            ckpt_path=ckpt_path,
            variant_name=variant_name,
            loaded_keys=loaded_keys,
            skipped_missing_keys=skipped_missing_keys,
            skipped_shape_keys=skipped_shape_keys,
        )
        raise RuntimeError(
            "Shape-mismatched encoder keys found during experimental SimCLR loading: "
            f"{audit}"
        )

    missing_keys_after, unexpected_keys_after = model.load_state_dict(loaded_sd, strict=False)

    disallowed_missing = [
        key for key in skipped_missing_keys if _get_prefix(key) not in allowed_missing_prefixes
    ]
    if disallowed_missing:
        audit = _make_audit(
            ckpt_path=ckpt_path,
            variant_name=variant_name,
            loaded_keys=loaded_keys,
            skipped_missing_keys=skipped_missing_keys,
            skipped_shape_keys=skipped_shape_keys,
        )
        raise RuntimeError(
            "Unexpected missing encoder keys found during experimental SimCLR loading: "
            f"{audit}"
        )

    for prefix in BACKBONE_ENCODER_PREFIXES:
        if loaded_by_prefix[prefix] == 0:
            audit = _make_audit(
                ckpt_path=ckpt_path,
                variant_name=variant_name,
                loaded_keys=loaded_keys,
                skipped_missing_keys=skipped_missing_keys,
                skipped_shape_keys=skipped_shape_keys,
            )
            raise RuntimeError(
                f"Required backbone encoder prefix '{prefix}' did not load any weights: {audit}"
            )

    for prefix in required_prefixes:
        if prefix not in BACKBONE_ENCODER_PREFIXES and loaded_by_prefix[prefix] == 0:
            audit = _make_audit(
                ckpt_path=ckpt_path,
                variant_name=variant_name,
                loaded_keys=loaded_keys,
                skipped_missing_keys=skipped_missing_keys,
                skipped_shape_keys=skipped_shape_keys,
            )
            raise RuntimeError(
                f"Required encoder prefix '{prefix}' did not load any weights: {audit}"
            )

    missing_required_target_keys = sorted(
        key for key in expected_target_encoder_keys if key not in loaded_sd
    )
    if missing_required_target_keys:
        audit = _make_audit(
            ckpt_path=ckpt_path,
            variant_name=variant_name,
            loaded_keys=loaded_keys,
            skipped_missing_keys=skipped_missing_keys,
            skipped_shape_keys=skipped_shape_keys,
            missing_required_target_keys=missing_required_target_keys,
        )
        raise RuntimeError(
            "Required target encoder keys were not fully loaded during experimental "
            f"SimCLR loading: {audit}"
        )

    audit = _make_audit(
        ckpt_path=ckpt_path,
        variant_name=variant_name,
        loaded_keys=loaded_keys,
        skipped_missing_keys=skipped_missing_keys,
        skipped_shape_keys=skipped_shape_keys,
        missing_required_target_keys=[],
    )
    audit["seen_candidate_encoder_key_count"] = encoder_seen
    audit["missing_keys_after_load"] = list(missing_keys_after)
    audit["unexpected_keys_after_load"] = list(unexpected_keys_after)

    if verbose:
        print(
            "[Pretrain->Seg Experimental] "
            f"variant={variant_name} | "
            f"loaded={audit['loaded_key_count']} | "
            f"skipped_missing={audit['skipped_missing_key_count']} | "
            f"skipped_shape={audit['skipped_shape_key_count']}"
        )

    return audit

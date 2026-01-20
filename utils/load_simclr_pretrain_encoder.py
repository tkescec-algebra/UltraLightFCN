import os
from typing import Any, Dict, Iterable, Tuple

import torch


# Encoder stack modules in UltraLightFCN:
# backbone blocks + mini_aspp + self-attention (sa)
ENC_PREFIXES: Tuple[str, ...] = (
    "block1",
    "dsconv2",
    "dsconv3",
    "dilconv4",
    "dilconv5",
    "mini_aspp",
    "sa",
)


def _is_state_dict_like(obj: Any) -> bool:
    """Heuristic: a state_dict is usually dict[str, Tensor]."""
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    for k, v in list(obj.items())[:5]:
        if not isinstance(k, str) or (not torch.is_tensor(v)):
            return False
    return True


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Extract a model state_dict from various checkpoint formats.

    Supported formats:
      - raw state_dict: dict[str, Tensor]
      - dict with known keys containing state_dict-like objects
      - nested dicts where a known key contains state_dict-like object
    """
    if _is_state_dict_like(ckpt):
        return ckpt

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    candidate_keys = (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
        "encoder",
        "backbone",
        "online_network",
        "student",
    )

    # Direct keys
    for key in candidate_keys:
        if key in ckpt and _is_state_dict_like(ckpt[key]):
            return ckpt[key]

    # One-level nested keys
    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            sub = ckpt[key]
            for key2 in candidate_keys:
                if key2 in sub and _is_state_dict_like(sub[key2]):
                    return sub[key2]

    # Fallback: search any value
    for _, v in ckpt.items():
        if _is_state_dict_like(v):
            return v

    raise RuntimeError(
        "Could not extract a state_dict from checkpoint. "
        f"Top-level keys: {list(ckpt.keys())[:30]}"
    )


def _strip_known_wrappers(key: str, wrappers: Iterable[str]) -> str:
    """Remove common wrapper prefixes repeatedly."""
    k = key
    changed = True
    while changed:
        changed = False
        for w in wrappers:
            if k.startswith(w):
                k = k[len(w) :]
                changed = True
    return k


def load_pretrained_encoder_into_ultralight(
    model: torch.nn.Module,
    ckpt_path: str,
    encoder_prefixes: Tuple[str, ...] = ENC_PREFIXES,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Load pretrained encoder weights into UltraLightFCN encoder modules.

    Policy:
      - only keys that start with encoder_prefixes are considered
      - only keys that exist in model.state_dict() AND have matching shapes are loaded
      - strict=False is used to allow partial loading (but we report stats)

    Returns a dict with loading statistics.
    """
    if (ckpt_path is None) or (not os.path.isfile(ckpt_path)):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    src_sd = _extract_state_dict(ckpt)
    tgt_sd = model.state_dict()

    wrappers = (
        "module.",
        "model.",
        "net.",
        "network.",
        "encoder.",
        "backbone.",
        "online_network.",
        "student.",
    )

    loaded_sd: Dict[str, torch.Tensor] = {}
    skipped_missing, skipped_shape, seen_encoder = 0, 0, 0

    for k, v in src_sd.items():
        k2 = _strip_known_wrappers(k, wrappers)

        if not k2.startswith(encoder_prefixes):
            continue

        seen_encoder += 1

        if k2 not in tgt_sd:
            skipped_missing += 1
            continue

        if tgt_sd[k2].shape != v.shape:
            skipped_shape += 1
            continue

        loaded_sd[k2] = v

    missing_keys, unexpected_keys = model.load_state_dict(loaded_sd, strict=False)

    stats = {
        "seen_encoder_keys_in_ckpt": int(seen_encoder),
        "loaded": int(len(loaded_sd)),
        "skipped_missing": int(skipped_missing),
        "skipped_shape": int(skipped_shape),
        "missing_keys_after": int(len(missing_keys)) if isinstance(missing_keys, list) else 0,
        "unexpected_keys_after": int(len(unexpected_keys)) if isinstance(unexpected_keys, list) else 0,
    }

    if verbose:
        print(
            "[Pretrain->Seg] "
            f"seen_encoder={stats['seen_encoder_keys_in_ckpt']} | "
            f"loaded={stats['loaded']} | "
            f"skipped_missing={stats['skipped_missing']} | "
            f"skipped_shape={stats['skipped_shape']}"
        )
        if stats["loaded"] == 0:
            print("[Pretrain->Seg] WARNING: loaded=0. Sample checkpoint keys:")
            for kk in list(src_sd.keys())[:30]:
                print("  ", kk)

    return stats

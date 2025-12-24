import os
import torch
from typing import Dict, Tuple, Any, Iterable


# These are your encoder module name prefixes in UltraLightFCN
ENC_PREFIXES = ("block1", "dsconv2", "dsconv3", "dilconv4", "dilconv5")


def _is_state_dict_like(obj: Any) -> bool:
    """Heuristic: a state_dict is usually a dict[str, Tensor]."""
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    # Check a few items
    for k, v in list(obj.items())[:5]:
        if not isinstance(k, str):
            return False
        if not torch.is_tensor(v):
            return False
    return True


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Try to extract a PyTorch state_dict from various checkpoint formats.
    Returns a dict[str, Tensor].
    """
    if _is_state_dict_like(ckpt):
        return ckpt

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    # Common container keys (ordered by likelihood)
    candidate_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
        "encoder",
        "backbone",
        "online_network",
        "student",
    ]

    for key in candidate_keys:
        if key in ckpt and _is_state_dict_like(ckpt[key]):
            return ckpt[key]

    # Sometimes nested: e.g. ckpt["model"]["state_dict"]
    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            sub = ckpt[key]
            for key2 in candidate_keys:
                if key2 in sub and _is_state_dict_like(sub[key2]):
                    return sub[key2]

    # Fallback: find the first dict[str,Tensor] inside ckpt
    for k, v in ckpt.items():
        if _is_state_dict_like(v):
            return v

    raise RuntimeError(
        "Could not extract a state_dict from checkpoint. "
        f"Top-level keys: {list(ckpt.keys())[:30]}"
    )


def _strip_prefixes(key: str, prefixes: Iterable[str]) -> str:
    """Remove any of the given prefixes repeatedly if present."""
    k = key
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                changed = True
    return k


def load_phase2_encoder_into_ultralight(
    model: torch.nn.Module,
    ckpt_path: str,
    encoder_prefixes: Tuple[str, ...] = ENC_PREFIXES,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Load Phase2 SimCLR weights into UltraLightFCN encoder modules.

    - Loads checkpoint from ckpt_path
    - Extracts state_dict robustly
    - Strips common wrappers (module., model., encoder., backbone., etc.)
    - Only copies parameters that:
        (a) belong to encoder blocks (by prefix), AND
        (b) exist in model.state_dict(), AND
        (c) have matching tensor shapes

    Returns stats dict with counts for reporting/logging.
    """
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    target = model.state_dict()

    # Prefixes we may want to remove (order doesn't matter due to loop)
    # This covers typical DDP / wrappers / SSL frameworks.
    strip_list = (
        "module.",
        "model.",
        "net.",
        "network.",
        "encoder.",
        "backbone.",
        "online_network.",
        "student.",
    )

    loaded, matched, skipped_shape, skipped_missing = 0, 0, 0, 0
    new_sd = {}

    for k, v in sd.items():
        k2 = _strip_prefixes(k, strip_list)

        # Only map encoder weights into UltraLightFCN encoder modules
        if not k2.startswith(encoder_prefixes):
            continue

        if k2 not in target:
            skipped_missing += 1
            continue

        if target[k2].shape != v.shape:
            skipped_shape += 1
            continue

        new_sd[k2] = v
        matched += 1

    # Load partial state dict
    missing_keys, unexpected_keys = model.load_state_dict(new_sd, strict=False)
    loaded = len(new_sd)

    if verbose:
        print(
            f"[Phase2->Seg] Loaded encoder keys: {loaded} | "
            f"matched: {matched} | skipped_missing: {skipped_missing} | skipped_shape: {skipped_shape}"
        )
        # Helpful when debugging a new checkpoint format
        if loaded == 0:
            sample_keys = list(sd.keys())[:30]
            print("[Phase2->Seg] WARNING: Loaded 0 keys. Sample checkpoint keys:")
            for kk in sample_keys:
                print("  ", kk)

    return {
        "loaded": loaded,
        "matched": matched,
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "missing_keys_after": len(missing_keys) if isinstance(missing_keys, list) else 0,
        "unexpected_keys_after": len(unexpected_keys) if isinstance(unexpected_keys, list) else 0,
    }

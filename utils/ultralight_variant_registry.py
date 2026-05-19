from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import torch

from models.UltraLightFCN_base import UltraLightFCN
from models.UltraLightFCN_experimental_variants import UltraLightFCNNoShallowSkip
from utils.config import SEG_PARAMS


VariantBuilder = Callable[..., torch.nn.Module]


@dataclass(frozen=True)
class UltraLightVariantSpec:
    variant_name: str
    params: dict[str, Any]
    model_class: type[torch.nn.Module]
    model_builder: VariantBuilder
    notes: list[str]
    simclr_compatibility_policy: dict[str, Any]


def _clone_seg_params() -> dict[str, Any]:
    return deepcopy(SEG_PARAMS)


def _build_standard_variant(
    params: dict[str, Any],
    in_channels: int = 3,
    num_classes: int = 1,
) -> UltraLightFCN:
    return UltraLightFCN(in_channels=in_channels, num_classes=num_classes, params=params)


def _build_no_shallow_skip_variant(
    params: dict[str, Any],
    in_channels: int = 3,
    num_classes: int = 1,
) -> UltraLightFCNNoShallowSkip:
    return UltraLightFCNNoShallowSkip(
        in_channels=in_channels,
        num_classes=num_classes,
        params=params,
    )


def _make_spec(
    variant_name: str,
    params: dict[str, Any],
    model_class: type[torch.nn.Module],
    model_builder: VariantBuilder,
    notes: list[str],
    removed_encoder_prefixes: list[str] | None = None,
    required_encoder_prefixes: list[str] | None = None,
    policy_name: str = "full_encoder_load_required",
) -> UltraLightVariantSpec:
    removed_prefixes = list(removed_encoder_prefixes or [])
    required_prefixes = list(
        required_encoder_prefixes
        or ["block1", "dsconv2", "dsconv3", "dilconv4", "dilconv5", "mini_aspp", "sa"]
    )
    return UltraLightVariantSpec(
        variant_name=variant_name,
        params=params,
        model_class=model_class,
        model_builder=model_builder,
        notes=notes,
        simclr_compatibility_policy={
            "policy_name": policy_name,
            "allowed_missing_prefixes": removed_prefixes,
            "required_encoder_prefixes": required_prefixes,
            "requires_backbone_load": True,
        },
    )


def get_ultralight_variant_spec(variant_name: str) -> UltraLightVariantSpec:
    params = _clone_seg_params()

    if variant_name == "baseline":
        return _make_spec(
            variant_name=variant_name,
            params=params,
            model_class=UltraLightFCN,
            model_builder=_build_standard_variant,
            notes=[
                "Canonical UltraLightFCN baseline cloned from utils.config.SEG_PARAMS.",
                "No modules removed; intended as the fixed reference architecture.",
            ],
        )

    if variant_name == "no_mini_aspp":
        params["mini_aspp"] = False
        return _make_spec(
            variant_name=variant_name,
            params=params,
            model_class=UltraLightFCN,
            model_builder=_build_standard_variant,
            notes=[
                "Removes the Mini-ASPP bottleneck context block.",
                "Keeps self-attention and the rest of the encoder-decoder topology unchanged.",
            ],
            removed_encoder_prefixes=["mini_aspp"],
            required_encoder_prefixes=["block1", "dsconv2", "dsconv3", "dilconv4", "dilconv5", "sa"],
            policy_name="partial_encoder_load_allowed_for_removed_modules",
        )

    if variant_name == "no_shifted_sa":
        params["use_sa"] = True
        params["sa_shifted"] = False
        return _make_spec(
            variant_name=variant_name,
            params=params,
            model_class=UltraLightFCN,
            model_builder=_build_standard_variant,
            notes=[
                "Keeps the self-attention block but disables shifted-window behavior.",
                "All encoder modules remain shape-compatible with the baseline SimCLR encoder.",
            ],
        )

    if variant_name == "no_mini_aspp_no_sa":
        params["mini_aspp"] = False
        params["use_sa"] = False
        return _make_spec(
            variant_name=variant_name,
            params=params,
            model_class=UltraLightFCN,
            model_builder=_build_standard_variant,
            notes=[
                "Removes both Mini-ASPP and self-attention from the encoder bottleneck.",
                "Retains the lightweight backbone and decoder structure.",
            ],
            removed_encoder_prefixes=["mini_aspp", "sa"],
            required_encoder_prefixes=["block1", "dsconv2", "dsconv3", "dilconv4", "dilconv5"],
            policy_name="partial_encoder_load_allowed_for_removed_modules",
        )

    if variant_name == "no_shallow_skip":
        return _make_spec(
            variant_name=variant_name,
            params=params,
            model_class=UltraLightFCNNoShallowSkip,
            model_builder=_build_no_shallow_skip_variant,
            notes=[
                "Removes the shallow skip projection and concatenation path after upconv7.",
                "Preserves the encoder, Mini-ASPP, self-attention, dilation, and output resolution.",
            ],
        )

    if variant_name == "no_dilation":
        params["dilations"] = [1, 1]
        return _make_spec(
            variant_name=variant_name,
            params=params,
            model_class=UltraLightFCN,
            model_builder=_build_standard_variant,
            notes=[
                "Keeps the dilated convolution stages but disables atrous expansion by using [1, 1].",
                "Tensor shapes stay unchanged relative to the baseline encoder.",
            ],
        )

    if variant_name == "decoder_narrow":
        params["dec_channels"] = [24, 12, 12]
        return _make_spec(
            variant_name=variant_name,
            params=params,
            model_class=UltraLightFCN,
            model_builder=_build_standard_variant,
            notes=[
                "Narrows the decoder channel widths while leaving the encoder intact.",
                "SimCLR loading remains fully encoder-compatible because only decoder widths change.",
            ],
        )

    if variant_name == "decoder_wide":
        params["dec_channels"] = [40, 20, 20]
        return _make_spec(
            variant_name=variant_name,
            params=params,
            model_class=UltraLightFCN,
            model_builder=_build_standard_variant,
            notes=[
                "Widens the decoder channel widths while leaving the encoder intact.",
                "SimCLR loading remains fully encoder-compatible because only decoder widths change.",
            ],
        )

    raise ValueError(f"Unknown UltraLightFCN experimental variant: {variant_name}")


def list_ultralight_variant_names() -> list[str]:
    return [
        "baseline",
        "no_mini_aspp",
        "no_shifted_sa",
        "no_mini_aspp_no_sa",
        "no_shallow_skip",
        "no_dilation",
        "decoder_narrow",
        "decoder_wide",
    ]


def build_ultralight_variant(
    variant_name: str,
    in_channels: int = 3,
    num_classes: int = 1,
) -> torch.nn.Module:
    spec = get_ultralight_variant_spec(variant_name)
    return spec.model_builder(
        params=deepcopy(spec.params),
        in_channels=in_channels,
        num_classes=num_classes,
    )


@torch.no_grad()
def run_ultralight_variant_smoke_test(
    input_shape: tuple[int, int, int, int] = (1, 3, 256, 256),
    expected_output_shape: tuple[int, int, int, int] = (1, 1, 256, 256),
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    dummy = torch.zeros(input_shape)

    for variant_name in list_ultralight_variant_names():
        try:
            model = build_ultralight_variant(variant_name=variant_name)
            model.eval()
            output = model(dummy)
            got_shape = tuple(output.shape)
            results[variant_name] = {
                "instantiated": True,
                "output_shape": got_shape,
                "shape_ok": got_shape == expected_output_shape,
            }
        except Exception as exc:  # pragma: no cover - diagnostic path
            results[variant_name] = {
                "instantiated": False,
                "output_shape": None,
                "shape_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return results


if __name__ == "__main__":
    smoke_results = run_ultralight_variant_smoke_test()
    for variant_name, result in smoke_results.items():
        print(f"{variant_name}: {result}")

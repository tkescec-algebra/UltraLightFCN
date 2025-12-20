import torch
from collections import OrderedDict
from ptflops import get_model_complexity_info

import segmentation_models_pytorch as smp

from models.UltraLightFCN_base import UltraLightFCN
from train.ablation import VARIANTS, build_ultralight_params, CHANNELS
from train.sota import SOTA_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_sota_model(cfg: dict) -> torch.nn.Module:
    """
    Kreira SOTA model na temelju konfiguracije iz SOTA_MODELS.
    Pretpostavka: koristiš segmentation_models_pytorch.
    """
    model_key = cfg["model_key"]
    encoder_name = cfg["encoder_name"]
    encoder_weights = cfg.get("encoder_weights", None)

    common_kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=CHANNELS,
        classes=1,          # binary mask
    )

    if model_key == "dlv3p":
        model = smp.DeepLabV3Plus(**common_kwargs)
    elif model_key == "unet":
        model = smp.Unet(**common_kwargs)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    return model


def profile_model(model: torch.nn.Module, name: str):
    model = model.to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    macs, nparams = get_model_complexity_info(
        model,
        (CHANNELS, 256, 256),
        as_strings=False,           # brojke kao float/int
        print_per_layer_stat=False,
        verbose=False
    )

    print(f"{name:20s}  MACs: {macs/1e9:.3f} G,  Params: {nparams/1e6:.3f} M")


def main():
    print("=== UltraLight ablation varijante ===")
    for variant_name, cfg in VARIANTS.items():
        params_dict = build_ultralight_params(cfg)
        model = UltraLightFCN(
            in_channels=CHANNELS,
            num_classes=1,
            params=params_dict
        )
        profile_model(model, variant_name)

    print("\n=== SOTA modeli ===")
    for model_name, cfg in SOTA_MODELS.items():
        model = build_sota_model(cfg)
        profile_model(model, model_name)


if __name__ == "__main__":
    main()

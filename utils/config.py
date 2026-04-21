ENCODER_PARAMS = {
    # Encoder
    'enc_channels': [16, 16, 32, 32, 64],
    'enc_kernel_sizes': [3, 3, 3, 3, 3],
    'enc_strides': [1, 2, 2, 1, 1],
    'dilations': [2, 4],
    # Context
    'mini_aspp': True,
    'mini_aspp_gpool': True,
    # Attention
    'use_sa': True,
    'sa_windowed': True,
    'sa_window_size': 16,
    'sa_shifted': True,
    'sa_heads': 4,
    'sa_dropout': 0.1,
}

SEG_PARAMS = dict(ENCODER_PARAMS)
SEG_PARAMS.update({
    # Decoder
    "dec_channels": [32, 16, 16],
    "dec_kernel_sizes": [3, 3],
    "dec_strides": [1, 1],
    "upscale": [2, 2],
})

ENCODER_PREFIXES: tuple[str, ...] = (
    "block1",
    "dsconv2",
    "dsconv3",
    "dilconv4",
    "dilconv5",
    "mini_aspp",
    "sa",
)


# -----------------------------
# Phase-7 plotting
# -----------------------------

# Desktop paths
PHASE7_MASTER_REPORT = r"bench_phase7/20260201_103832/phase7_master_report.json"
PHASE7_PLOTS_OUTDIR = r"bench_phase7/overall_plots"

# Edge device paths
PHASE7_MASTER_REPORT_JETSON = r"bench_phase7_jetson_ts/20260308_103016/phase7_master_report.json"
PHASE7_PLOTS_OUTDIR_JETSON = r"bench_phase7_jetson_ts/overall_plots"

# Paper-safe default: ours + SOTA MINFT only (fullft is optional appendix)
PHASE7_INCLUDE_FULLFT = False

PHASE7_MODEL_ORDER = [
    "ultralight_phase6",
    "dlv3p_resnet50",
    "dlv3p_mobilenetv2",
    "unet_resnet34",
]

PHASE7_MODEL_LABELS = {
    "ultralight_phase6": "ULFCN (ours)",
    "dlv3p_resnet50": "DLV3+ R50",
    "dlv3p_mobilenetv2": "DLV3+ MNetV2",
    "unet_resnet34": "U-Net R34",
}

# -----------------------------
# Phase-7 per-image plots (.npz)
# -----------------------------
# Desktop paths
PHASE7_PER_IMAGE_DIR = r"bench_phase7/20260201_103832/per_image"
PHASE7_PER_IMAGE_OUTDIR = r"bench_phase7/plots_per_image"

# Edge device paths
PHASE7_PER_IMAGE_DIR_JETSON = r"bench_phase7_jetson_ts/20260308_103016/per_image"
PHASE7_PER_IMAGE_OUTDIR_JETSON = r"bench_phase7_jetson_ts/plots_per_image"

# Tail threshold for "failure rate" plots
PHASE7_TAIL_DICE_THRESH = 0.10

# How many worst images to export (per model / overall)
PHASE7_WORST_N = 50

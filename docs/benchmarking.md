# Benchmarking And Deployment Guide

## Phase 7 Desktop Benchmark

Script: `test/phase7_test_benchmark.py`

Default desktop config consumes final checkpoint rosters from:

- `../train/seg_phase6/final_retrain90/phase6_test_report.json`
- `../train/seg_sota/stage2_final90_test/phaseSOTA_test_report.json`

It uses `../dataset/test`, `input_size = 256`, threshold `0.5`, deterministic TEST preprocessing, and output root `bench_phase7`.

Outputs include:

- `config.json`
- `phase7_master_report.json`
- `phase7_quality_summary.csv`
- `phase7_timing_per_repeat.csv`
- `phase7_timing_aggregate.csv`
- `per_image/*.npz` when per-image metrics are enabled

Metrics and reporting include hard Dice, IoU, precision, recall, subgroup reporting for `PV01`/`PV03`/`PV08` plus `OTHER`, parameter counts, optional FLOPs/MACs, CPU timing, and GPU timing when CUDA is available.

## TorchScript Export

Script: `tools/export_torchscript_phase7_torch10.py`

Default rosters:

- `train/seg_phase6/final_retrain90/phase6_test_report.json`
- `train/seg_sota/stage2_final90_test/phaseSOTA_test_report.json`

Output folder:

- `tools/export_torchscript_10`

The exporter handles UltraLightFCN and SOTA SMP models. UltraLightFCN export uses `models/UltraLightFCN_base_deploy.py`, which contains TorchScript-friendly replacements and comments about PyTorch 1.10 compatibility. By default, SOTA export includes `minft` models and skips `fullft` unless `SOTA_INCLUDE_FULLFT=1`.

Environment caveat: the exporter comments say it is designed for Python 3.6 plus torch 1.10.x to match Jetson Nano/JetPack 4.x, while the main `requirements.txt` targets Torch 2.7.1 CUDA 12.8.

## Jetson TorchScript Benchmark

Script: `test/phase7_jetson_torchscript_benchmark.py`

Default config:

- `ts_root = "tools/export_torchscript_10"`
- `data_root = "dataset"`
- `test_split = "test"`
- `out_root = "bench_phase7_jetson_ts"`

The Jetson/TorchScript benchmark reimplements desktop-compatible TEST preprocessing without Albumentations, loads `.ts` files, and isolates each model in a subprocess. The subprocess design is explicitly for robustness on memory-constrained Jetson devices: failures and OOM-like exits are recorded per model instead of crashing the entire benchmark.

Outputs include the same core benchmark reports as desktop plus model status files:

- `phase7_model_status.csv`
- `phase7_model_status.jsonl`
- per-worker stdout/stderr/result files under `workers/`

## Plot Generation

Plot scripts are present:

- `test/plot_phase7_overall.py`
- `test/plot_phase7_overall_jetson.py`
- `train/plot_training_validation_loss_mean_sd.py`
- `pretrain/aug_ablation/make_aug_heatmaps.py`
- `pretrain/aug_ablation/make_fig3_simclr.py`
- `optuna_study/make_supp_optuna_simclr.py`

Hardcoded timestamp caveat: `utils/config.py` points Phase 7 plotting at:

- `bench_phase7/20260201_103832/phase7_master_report.json`
- `bench_phase7_jetson_ts/20260308_103016/phase7_master_report.json`

Update these config constants or provide matching artifact paths before regenerating plots.

## Artifact-Assisted Benchmarking

To benchmark without rerunning all phases, users need:

- Prepared `dataset/test` with image/mask pairs.
- Phase 6 final report and LAST checkpoints for UltraLightFCN.
- SOTA final reports and LAST checkpoints if SOTA comparisons are included.
- TorchScript `.ts` files and metadata under `tools/export_torchscript_10` for Jetson/TorchScript benchmarking.

For plot-only reproduction, provide the relevant `phase7_master_report.json`, timing CSVs, quality CSVs, and per-image artifacts expected by the plotting scripts/config.

## Known Benchmark Risks

- Phase 7 plotting paths in `utils/config.py` are timestamp-specific.
- `tools/check_all_torchscript_models.py` uses `/work/tools/export_torchscript_10`, which is an absolute/container-specific path.
- `test/smoke_ts.py` points at `export_torchscript/dlv3p_mobilenetv2_fullft_seed_13__seed13.ts`, while the current exporter writes to `tools/export_torchscript_10`.
- Main training requirements use Torch 2.7.1 CUDA 12.8; the exporter is documented for torch 1.10.x deployment compatibility.
- Desktop and Jetson benchmark scripts use different relative path assumptions (`../dataset` from `test/` versus `dataset` from repository root).

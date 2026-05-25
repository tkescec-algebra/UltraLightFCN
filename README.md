# UltraLightFCN

## Project Overview

UltraLightFCN is a phase-based research repository for lightweight binary semantic segmentation. The current workflow uses a PV dataset testbed, SimCLR encoder pretraining, staged model selection, locked-box TEST evaluation, and deployment benchmarking.

The repository contains model code, data preprocessing, hyperparameter optimization, pretraining, final segmentation training, optional comparator/ablation branches, and desktop/Jetson benchmarking utilities. It is not currently packaged as an installable Python package.

## Repository Map

| Path | Role |
| --- | --- |
| `models/` | UltraLightFCN, SimCLR, deployment, and experimental architecture model definitions. |
| `utils/` | Shared config, datasets, transforms, losses, metrics, reproducibility, registries, and loading helpers. |
| `preprocessing/` | Raw BMP to prepared PNG dataset pipeline and dataset figure code. |
| `optuna_study/` | SimCLR and segmentation Optuna HPO scripts and local Optuna artifacts. |
| `pretrain/` | SimCLR top-K retraining, final pretraining, UMAP, and augmentation ablation code. |
| `train/` | Segmentation retraining, final TEST evaluation, SOTA comparators, ablations, and statistics. |
| `test/` | Phase 7 desktop/TorchScript benchmarks, plotting, and BDS analysis. |
| `tools/` | TorchScript export/check utilities. |
| `data/`, `temp/`, `dataset/` | Local raw data, preprocessing cache, and prepared split dataset. These are not source. |

## Installation

Install dependencies from `requirements.txt` in a suitable Python environment. The file pins PyTorch packages from the CUDA 12.8 wheel index:

```powershell
pip install -r requirements.txt
```

The main environment is built around `torch==2.7.1+cu128`, `torchvision==0.22.1+cu128`, `torchaudio==2.7.1+cu128`, `albumentations`, `opencv`, `optuna`, `segmentation_models_pytorch`, `ptflops`, and reporting/scientific packages.

TorchScript export/deployment is a separate compatibility story: `tools/export_torchscript_phase7_torch10.py` says it is designed for Python 3.6 plus torch 1.10.x to match Jetson Nano/JetPack 4.x constraints.

## Dataset Layout

Raw data defaults to `data/` when preprocessing is launched from `preprocessing/` via `../data`. The expected raw files are recursive BMP image/mask pairs:

- Image: `<base>.bmp`
- Mask: `<base>_label.bmp`

Prepared data defaults to `dataset/` with split folders:

- `dataset/train`
- `dataset/valid`
- `dataset/test`

Each prepared image is a PNG with a matching `<stem>_label.png` mask in the same split folder. See [docs/data.md](docs/data.md).

## Quickstart / Minimal Framework Usage

The reusable pieces are:

- Model definitions: `models/UltraLightFCN_base.py`, `models/UltraLightFCN_SimCLR.py`, `models/UltraLightFCN_base_deploy.py`.
- Default architecture parameters and plotting paths: `utils/config.py`.
- Segmentation and SimCLR datasets: `utils/dataset.py`.
- Train/valid/test transforms: `utils/transforms.py`.
- Losses and metrics: `utils/loss_functions.py`, `utils/metrics.py`.
- Reproducibility helpers: `utils/repro.py`.

Imports are path-sensitive because scripts are written as research entrypoints with relative paths. Prefer following the phase scripts instead of assuming package-style imports.

## Full Reproduction Overview

| Phase | Script | Purpose | Expensive |
| --- | --- | --- | --- |
| 0 | `preprocessing/image_preprocessing.py` | Prepare `dataset/` from raw BMP pairs. | Yes |
| 1 | `optuna_study/phase1_simclr_study_UltraLightFCN.py` | SimCLR HPO. | Yes |
| 2 | `pretrain/phase2_simclr_retrain_top10_downstream.py` | Top-K SimCLR retrain and downstream selection. | Yes |
| 3 | `pretrain/phase3_simclr_full_pretrain.py` | Final SimCLR pretraining on TRAIN. | Yes |
| 4 | `optuna_study/phase4_seg_study_UltraLightFCN.py` | Segmentation HPO using Phase 3 encoder. | Yes |
| 5 | `train/phase5_seg_retrain_topk.py` | Top-K segmentation confirmation and winner JSON. | Yes |
| 6 | `train/phase6_seg_final_retrain90_test.py` | Final TRAIN+VALID retrain and locked-box TEST. | Yes |
| 7 | `test/phase7_test_benchmark.py`, `tools/export_torchscript_phase7_torch10.py`, `test/phase7_jetson_torchscript_benchmark.py` | Desktop benchmark, TorchScript export, Jetson/TorchScript benchmark. | Yes |

See [docs/reproduction.md](docs/reproduction.md). There is no one-command reproduction wrapper in the current source.

## Comparators And Optional Experiments

Present optional branches include:

- SOTA comparators via `train/phaseSOTA_stage1_dev80.py`, `train/phaseSOTA_stage2_final90_test.py`, and `utils/sota_registry.py`.
- no-SimCLR ablations via no-SimCLR Phase 4/5/6 scripts.
- SimCLR augmentation ablation under `pretrain/aug_ablation/`.
- UMAP, statistical, plotting, and BDS analysis scripts.

See [docs/experiments.md](docs/experiments.md).

## Benchmarking And Deployment

Phase 7 evaluates final checkpoints on the locked TEST split, records quality and timing, exports TorchScript models, and benchmarks `.ts` models with subprocess isolation for memory-constrained Jetson-style runs. See [docs/benchmarking.md](docs/benchmarking.md).

## Artifact Policy

Generated artifacts include prepared datasets, temp caches, Optuna DBs, checkpoints, logs, benchmark outputs, TorchScript exports, figures, and paper-facing CSV/JSON summaries. Some generated summaries may be useful release artifacts, but this repository does not treat generated folders as source. See [docs/artifacts.md](docs/artifacts.md).

## Known Limitations / Repo Assumptions

- Many scripts use relative paths such as `../dataset`, so launch directory matters.
- Reproducing from scratch is expensive because HPO, pretraining, segmentation retraining, final TEST evaluation, and benchmarking are separate phases.
- Artifact-assisted reproduction requires externally provided checkpoints/reports if users skip expensive phases.
- Phase 7 plotting defaults in `utils/config.py` contain hardcoded timestamped output paths.
- TorchScript export targets an older deployment environment than the main training requirements.
- `tools/check_all_torchscript_models.py` and `test/smoke_ts.py` have path assumptions that differ from the current exporter output folder.

## Citation / License / Contact

Citation, license, and contact information are not defined in the current repository source. Add final publication and licensing details here when available.

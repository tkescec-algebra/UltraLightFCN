# PROJECT_DISCOVERY_REPORT

## 1. Executive summary
This repository implements an end-to-end experimental pipeline for solar-panel binary semantic segmentation centered on a custom lightweight model, `UltraLightFCN`, with optional SimCLR self-supervised pretraining and Phase-7 deployment benchmarking. The codebase is organized around explicit methodology phases:

| Phase | Purpose | Primary files |
| --- | --- | --- |
| Preprocessing | Convert raw BMP image/mask pairs into tiled PNG train/valid/test splits | `preprocessing/image_preprocessing.py` |
| Phase 1 | SimCLR HPO on a reduced pretraining pool | `optuna_study/phase1_simclr_study_UltraLightFCN.py` |
| Phase 2 | Retrain top-K SimCLR candidates and select encoder via downstream mini-seg VALID warm-up | `pretrain/phase2_simclr_retrain_top10_downstream.py` |
| Phase 3 | Full SimCLR pretraining on downstream TRAIN only | `pretrain/phase3_simclr_full_pretrain.py` |
| Phase 4 | Segmentation HPO using Phase-3 encoder init | `optuna_study/phase4_seg_study_UltraLightFCN.py` |
| Phase 5 | Top-K segmentation retraining / confirmation on full TRAIN+VALID protocol | `train/phase5_seg_retrain_topk.py` |
| Phase 6 | Final retraining on TRAIN+VALID and locked-box TEST evaluation | `train/phase6_seg_final_retrain90_test.py` |
| Phase 7 | Desktop TEST benchmarking, TorchScript export, Jetson benchmarking, reporting plots | `test/phase7_test_benchmark.py`, `tools/export_torchscript_phase7_torch10.py`, `test/phase7_jetson_torchscript_benchmark.py` |
| SOTA comparator | SMP-based baseline training and final TEST evaluation | `train/phaseSOTA_stage1_dev80.py`, `train/phaseSOTA_stage2_final90_test.py` |

Confirmed dataset state in the working tree:

| Split | Images | Masks | Source |
| --- | ---: | ---: | --- |
| `dataset/train` | 22,577 | 22,577 | filesystem count |
| `dataset/valid` | 2,995 | 2,995 | filesystem count |
| `dataset/test` | 2,825 | 2,825 | filesystem count |

Observed subgroup distribution by filename prefix:

| Split | PV01 | PV03 | PV08 | OTHER |
| --- | ---: | ---: | ---: | ---: |
| `train` | 516 | 17,466 | 4,595 | 0 |
| `valid` | 64 | 2,414 | 517 | 0 |
| `test` | 65 | 2,244 | 516 | 0 |

## 2. Repository map
### Top-level folders

| Path | Role |
| --- | --- |
| `data/` | Raw source imagery and masks, stored in nested folders, original `.bmp` pairs. |
| `temp/` | Intermediate normalized PNG cache created by preprocessing. |
| `dataset/` | Final flat split folders `train/`, `valid/`, `test/` used by training/eval. |
| `models/` | Custom model definitions for UltraLightFCN, SimCLR encoder/head, and deploy-safe variant. |
| `preprocessing/` | Dataset construction and descriptive figure scripts. |
| `optuna_study/` | HPO entrypoints for SimCLR and segmentation, plus study plotting. |
| `pretrain/` | Phase-2/3 pretraining logic, UMAP analysis, augmentation ablation utilities. |
| `train/` | Downstream training phases, SOTA baseline stages, ablation, plotting. |
| `test/` | Phase-7 benchmark scripts, plotters, Jetson benchmark variant, BDS sensitivity analysis. |
| `tools/` | TorchScript export and verification tools. |
| `utils/` | Shared config, datasets, transforms, losses, metrics, reproducibility, helpers. |

### Key files outside folders

| Path | Role |
| --- | --- |
| `requirements.txt` | Environment specification pinned to `torch==2.7.1+cu128` and supporting libraries. |
| `visualize_overlays.py` | Standalone overlay visualization utility. |

## 3. End-to-end pipeline reconstruction
### Chronological pipeline
1. Raw data ingestion starts from `data/` in `preprocessing/image_preprocessing.py`.
2. Step 0/1 converts only missing raw BMP image/mask pairs into normalized PNG pairs under `temp/`.
3. Step 2 tiles images at `256x256`, keeps all positive tiles, and mines a bounded number of negatives per parent image and subset.
4. Step 3 performs group-aware splitting by original parent image within subsets `PV01`, `PV03`, `PV08`, then writes flat PNG image/mask pairs into `dataset/train`, `dataset/valid`, and `dataset/test`.
5. Phase 1 (`optuna_study/phase1_simclr_study_UltraLightFCN.py`) runs SimCLR HPO on a reduced deterministic file pool from `dataset/train`, with an internal pretrain train/val split and Optuna objective `avg_last_k` validation NT-Xent ratio.
6. Phase 2 (`pretrain/phase2_simclr_retrain_top10_downstream.py`) loads the top-K completed Phase-1 trials, retrains each SimCLR candidate on the full downstream TRAIN split, then selects the encoder using a short downstream segmentation warm-up evaluated on official `dataset/valid`.
7. Phase 3 (`pretrain/phase3_simclr_full_pretrain.py`) reloads the best Phase-2 hyperparameters and performs the final full-budget SimCLR pretraining on `dataset/train` only, saving the final epoch checkpoint as the transfer source.
8. Phase 4 (`optuna_study/phase4_seg_study_UltraLightFCN.py`) initializes UltraLightFCN from the Phase-3 LAST encoder checkpoint and performs segmentation HPO on TRAIN with official VALID evaluation.
9. Phase 5 (`train/phase5_seg_retrain_topk.py`) reloads the top-K Phase-4 trials, retrains each candidate on full TRAIN and full VALID across three seeds, then selects a winner by mean `best_avg_last_k_soft` across seeds.
10. Phase 6 (`train/phase6_seg_final_retrain90_test.py`) retrains the Phase-5 winner on TRAIN+VALID, saves LAST checkpoints only, and performs a single locked-box TEST evaluation aggregated across seeds.
11. SOTA Stage 1 (`train/phaseSOTA_stage1_dev80.py`) trains SMP baselines on TRAIN/VALID using the same unified recipe envelope derived from the Phase-5 winner.
12. SOTA Stage 2 (`train/phaseSOTA_stage2_final90_test.py`) retrains those baselines on TRAIN+VALID and performs locked-box TEST evaluation.
13. Phase 7 desktop benchmarking (`test/phase7_test_benchmark.py`) consumes final Phase-6 and SOTA Stage-2 checkpoints, recomputes TEST metrics deterministically, measures params/FLOPs/latency/memory, and emits per-image artifacts plus master reports.
14. TorchScript export (`tools/export_torchscript_phase7_torch10.py`) converts final checkpoints into `.ts` files aimed at Torch 1.10 / Jetson Nano compatibility.
15. Jetson benchmarking (`test/phase7_jetson_torchscript_benchmark.py`) benchmarks the exported TorchScript models in isolated subprocesses to survive OOM failures.

### Manuscript-aligned phase naming
Confirmed phase naming appears directly in filenames and docstrings:

| Methodology phase | Script(s) |
| --- | --- |
| Phase 1 SimCLR study | `optuna_study/phase1_simclr_study_UltraLightFCN.py` |
| Phase 2 top-K retrain + downstream selection | `pretrain/phase2_simclr_retrain_top10_downstream.py` |
| Phase 3 final SimCLR pretraining | `pretrain/phase3_simclr_full_pretrain.py` |
| Phase 4 segmentation HPO | `optuna_study/phase4_seg_study_UltraLightFCN.py` |
| Phase 5 top-K confirmation retrain | `train/phase5_seg_retrain_topk.py` |
| Phase 6 final retrain90 + TEST | `train/phase6_seg_final_retrain90_test.py` |
| Phase 7 benchmark/export/reporting | `test/phase7_test_benchmark.py`, `test/phase7_jetson_torchscript_benchmark.py`, `tools/export_torchscript_phase7_torch10.py` |

## 4. Model architecture discovery
### UltraLightFCN base model
Confirmed from `models/UltraLightFCN_base.py`:

| Stage | Structure |
| --- | --- |
| `block1` | Standard `Conv2d -> BatchNorm2d -> ReLU` |
| `dsconv2`, `dsconv3` | Depthwise separable convolution blocks with BN/ReLU |
| `dilconv4`, `dilconv5` | Dilated convolutions with dilation factors from config, BN/ReLU |
| `mini_aspp` | Lightweight context block with 1x1 branch, two dilated depthwise-separable branches, optional global pooling branch, then 1x1 fusion |
| `sa` | Windowed shifted self-attention bottleneck with LayerNorm, Q/K/V projections, residual MLP, and optional shifted-window masking |
| Decoder | Two bilinear upsample + conv blocks, one shallow skip projected from `block1`, concat fusion, final `1x1` logits |

Default configured architecture used in experiments is sourced from `utils/config.py`, not the internal defaults in `models/UltraLightFCN_base.py`:

| Parameter group | Configured values in `utils/config.py` |
| --- | --- |
| Encoder channels | `[16, 16, 32, 32, 64]` |
| Encoder strides | `[1, 2, 2, 1, 1]` |
| Dilations | `[2, 4]` |
| Decoder channels | `[32, 16, 16]` |
| Mini-ASPP | enabled, `mini_aspp_gpool=True` |
| Self-attention | enabled, windowed, shifted, `window_size=16`, `heads=4`, `dropout=0.1` |

### SimCLR variant
Confirmed from `models/UltraLightFCN_SimCLR.py`:

| Component | Role |
| --- | --- |
| `UltraLightEncoder` | Reuses the encoder + bottleneck context/attention stack and returns `(deep_feature, shallow_skip)` |
| `ProjectionHead` | Two-layer MLP `Linear -> ReLU -> Linear` |
| `SimCLRModel` | `encoder -> GAP -> projection head -> L2 normalization` |
| `UltraLightSegmentation` | Fine-tuning model that reuses the pretrained encoder and adds a lightweight decoder analogous to the base model |

### Deployment / TorchScript variant
`models/UltraLightFCN_base_deploy.py` is a deploy-safe variant of the same architecture. Confirmed differences:

| Difference | Rationale |
| --- | --- |
| Attention path uses manual scaled dot-product attention instead of `F.scaled_dot_product_attention` | TorchScript / older PyTorch compatibility |
| Attention mask construction is rewritten with explicit indexing | TorchScript-friendly control flow |
| Otherwise structure mirrors the base model | Intended for deployment equivalence |

## 5. Data and preprocessing assumptions
### Raw data assumptions
Confirmed from `preprocessing/image_preprocessing.py`:

| Assumption | Source |
| --- | --- |
| Raw inputs are recursive `.bmp` files under `../data` | `in_dir = Path("../data")` |
| Masks share folder with image and are named `<base>_label.bmp` | `collect_pairs_recursive_bmp()` |
| Subset ID is inferred from filename prefix `PV01_`, `PV03_`, `PV08_` | `get_subset_from_base()` |

### Prepared dataset assumptions
Confirmed from `utils/dataset.py` and observed filesystem:

| Assumption | Details |
| --- | --- |
| Final split folders are flat directories | `dataset/train`, `dataset/valid`, `dataset/test` |
| Image filenames exclude mask suffix | masks must not end with `_label.png` |
| Mask filenames are `<stem>_label.png` by default | `mask_suffix="_label"`, `mask_ext=".png"` |
| Images are read as RGB | `cv2.imread(..., IMREAD_COLOR)` then `cv2.cvtColor(..., BGR2RGB)` |
| Masks are read grayscale and binarized by `> 0` | `cv2.IMREAD_GRAYSCALE`, then cast to float |

### Tiling, positives, negatives, and splits
Confirmed from `preprocessing/image_preprocessing.py`:

| Rule | Value |
| --- | --- |
| Tile size | `256` |
| Stride | `256` |
| Positive tile by coverage | `coverage >= 0.005` |
| Positive tile by absolute pixels | `pos_pixels >= 64` |
| Per-parent negative quota | `min(alpha * P, max_neg_per_parent, len(cands))` with `alpha=2.0`, `max_neg_per_parent=4` |
| Hard-negative fraction | `0.5` |
| Empty-parent top-up | `global_empty_alpha=0.2`, `global_empty_cap=5000` |
| Split ratios | `(0.8, 0.1, 0.1)` |

Leakage prevention confirmed in preprocessing:

| Mechanism | Effect |
| --- | --- |
| Grouping by `parent_id` in `split_by_parent_within_subset()` | all tiles from one original image stay in one split |
| Subset-specific splitting | `PV01`, `PV03`, `PV08` are split independently |
| Dataset loaders exclude mask files explicitly | prevents image/mask leakage into file lists |

### Augmentation and preprocessing rules
Segmentation transforms from `utils/transforms.py`:

| Mode | Geometry | Photometric |
| --- | --- | --- |
| `train` | `LongestMaxSize -> PadIfNeeded`, horizontal flip, vertical flip, `RandomRotate90`, affine, mild grid distortion | hue/saturation/value, brightness/contrast, Gaussian noise, optional blur, ImageNet normalization |
| `valid` / `test` | deterministic `LongestMaxSize -> PadIfNeeded` only | ImageNet normalization |

SimCLR transforms from `pretrain/utils/simclr_transforms.py`:

| Transform | Default |
| --- | --- |
| Crop | `RandomResizedCrop(256, scale=(0.4,1.0))` |
| Flips | horizontal `0.5`, vertical `0.5` |
| Rotation | `10` degrees |
| Color jitter | `p=0.8` |
| Grayscale | `p=0.1` |
| Blur | `p=0.5`, kernel `3` |

## 6. Training and selection workflow
### Phase 1: SimCLR HPO
Confirmed from `optuna_study/phase1_simclr_study_UltraLightFCN.py`:

| Aspect | Behavior |
| --- | --- |
| Data source | `../dataset/train` only |
| Pool reduction | deterministic reduced file list, `reduce_max_total=5120` |
| Pretrain val split | `pretrain_val_frac=0.10`, stratified by subset prefix |
| Tuned params | `simclr_lr`, `simclr_temperature`, `weight_decay`, `warmup_ratio`, `proj_hidden_dim`, `proj_out_dim`, `max_grad_norm` |
| Objective | minimize average of last-K validation NT-Xent ratios |
| Scheduler | `timm.scheduler.CosineLRScheduler` |
| Study backend | `sqlite:///UltraLightFCN_study.db` |

### Phase 2: SimCLR top-K retrain and downstream-aware selection
Confirmed from `pretrain/phase2_simclr_retrain_top10_downstream.py`:

| Aspect | Behavior |
| --- | --- |
| Input study | `../optuna_study/UltraLightFCN_study.db`, study `UltraLightFCN_SimCLR_pretrain_RGB` |
| Candidate count | `topk = 10` |
| Retrain data | full downstream `../dataset/train` |
| Selection data | official `../dataset/valid` |
| Selection proxy | short frozen-encoder mini segmentation warm-up; selected by `mini_val_soft_dice` |
| Diagnostics | SimCLR alignment/uniformity logged but not used for selection |
| Output summary | `pretrain/checkpoints/simclr_topk_retrain_downstream/phase2_topk_results.csv` |

### Phase 3: final SimCLR pretraining
Confirmed from `pretrain/phase3_simclr_full_pretrain.py`:

| Aspect | Behavior |
| --- | --- |
| Hyperparameter source | best row from Phase-2 CSV |
| Training split | `../dataset/train` only |
| Seeds | default `(13,)` |
| Epochs | `200` |
| Checkpoint policy | LAST only, no proxy validation selection |
| Output root | `pretrain/checkpoints/simclr_phase3` |

### Phase 4: segmentation HPO
Confirmed from `optuna_study/phase4_seg_study_UltraLightFCN.py`:

| Aspect | Behavior |
| --- | --- |
| Initialization | Phase-3 LAST encoder checkpoint |
| Fixed model architecture | `SEG_PARAMS` only; HPO does not change architecture |
| TRAIN subset | optional fixed subset list, default `20%` of TRAIN |
| VALID evaluation | full VALID by default |
| Tuned params | `batch_size`, `base_lr`, `enc_lr_mult`, `weight_decay`, `rlop_factor`, `rlop_patience`, plus loss weights |
| Loss search | `BCEDiceLoss` or `BCEDiceFocalLoss` |
| Objective | maximize `best_avg_last_k` soft Dice |

### Phase 5: top-K retraining / confirmation
Confirmed from `train/phase5_seg_retrain_topk.py`:

| Aspect | Behavior |
| --- | --- |
| Candidate source | top-K complete Phase-4 trials |
| Splits used | full TRAIN + full VALID |
| Seeds | `(13, 37, 71)` |
| Epochs | `30` |
| Winner selection | highest mean `best_avg_last_k_soft` across seeds, tie-break lower std, then higher `best_val_soft` |
| Output artifacts | `seg_phase5/topk_retrain/phase5_topk_results.csv`, `seg_phase5/topk_retrain/phase5_winner.json` |

### Phase 6: final train+valid retraining
Confirmed from `train/phase6_seg_final_retrain90_test.py`:

| Aspect | Behavior |
| --- | --- |
| Training data | `ConcatDataset([train(mode="train"), valid(mode="train")])` |
| Seeds | `(13, 37, 71)` |
| Epochs | `60` |
| Scheduler | `CosineAnnealingLR` |
| Checkpoint policy | LAST only |
| TEST usage | only after training, once per seed |
| Output artifacts | `seg_phase6/final_retrain90/phase6_test_per_seed.csv`, `seg_phase6/final_retrain90/phase6_test_report.json` |

## 7. TEST evaluation and benchmarking workflow
### Locked-box TEST evaluation
Phase-6 and SOTA Stage-2 both follow the same pattern:

| Workflow | Files |
| --- | --- |
| Train on TRAIN+VALID, never TEST | `train/phase6_seg_final_retrain90_test.py`, `train/phaseSOTA_stage2_final90_test.py` |
| Evaluate on TEST once per trained seed | same files |
| Aggregate mean and population std across seeds | same files |

### Metric implementation
Confirmed in `utils/metrics.py`:

| Metric | Implementation |
| --- | --- |
| Soft Dice | `sigmoid(logits)` with no threshold |
| Hard Dice | `sigmoid(logits) > thr` |
| Soft IoU | same convention, no threshold |
| Hard IoU | thresholded at provided `thr` |
| Precision/Recall | thresholded, computed per image then batch-averaged |

Standard threshold in downstream evaluation and Phase-7 reporting is `0.5`.

### Phase-7 desktop benchmark
Confirmed from `test/phase7_test_benchmark.py`:

| Output / behavior | Details |
| --- | --- |
| Roster sources | `../train/seg_phase6/final_retrain90/phase6_test_report.json`, `../train/seg_sota/stage2_final90_test/phaseSOTA_test_report.json` |
| Quality recompute | deterministic, CPU-side TEST recomputation |
| Per-image artifact | compressed `.npz` with `dice`, `precision`, `recall`, optional `iou`, optional `group` |
| Subgroup summaries | overall plus `PV01`, `PV03`, `PV08`, `OTHER` |
| Efficiency | total params, encoder params, decoder params, FLOPs, MACs |
| Timing | per-repeat `ms/img`, FPS, RSS, VRAM alloc/reserved, across CPU and GPU |
| Composite score | `hardDice@0.5 / log10(1 + Params)` |
| Master report | timestamped `bench_phase7/<run>/phase7_master_report.json` |

### Jetson benchmark
Confirmed from `test/phase7_jetson_torchscript_benchmark.py`:

| Feature | Behavior |
| --- | --- |
| Input model format | TorchScript `.ts` |
| Dataset preprocessing | reimplemented deterministic test geometry without Albumentations |
| Fault tolerance | one model per subprocess to isolate OOM crashes |
| Intended target | Jetson Nano / Torch 1.10-era deployment |

## 8. Baseline/SOTA comparator workflow
### Comparator inventory
Confirmed from `utils/sota_registry.py`:

| Model ID | Architecture |
| --- | --- |
| `dlv3p_resnet50` | `segmentation_models_pytorch.DeepLabV3Plus` with `resnet50` encoder |
| `dlv3p_mobilenetv2` | `segmentation_models_pytorch.DeepLabV3Plus` with `mobilenet_v2` encoder |
| `unet_resnet34` | `segmentation_models_pytorch.Unet` with `resnet34` encoder |

Two fine-tuning regimes are defined:

| Regime | Meaning |
| --- | --- |
| `minft` | encoder LR multiplier fixed to `0.1` |
| `fullft` | encoder LR multiplier inherited from the Phase-5 winner |

### Fairness controls
Confirmed in `train/phaseSOTA_stage1_dev80.py` and `train/phaseSOTA_stage2_final90_test.py`:

| Control | Implementation |
| --- | --- |
| Shared batch size / base LR / weight decay / loss recipe | loaded from `seg_phase5/topk_retrain/phase5_winner.json` |
| Shared TRAIN/VALID or TRAIN+VALID split policy | same splits as UltraLightFCN phases |
| Shared seed set | `(13, 37, 71)` |
| Same metrics | `calculate_dice`, `calculate_iou`, `calculate_precision_recall` |
| Same checkpoint policy at final stage | LAST only |
| Same locked-box TEST principle | Stage 2 only evaluates after training |

## 9. Configuration and reproducibility
### Important configuration sources

| Concern | Source |
| --- | --- |
| Canonical UltraLightFCN experiment params | `utils/config.py` |
| Raw/preprocessed dataset paths | `preprocessing/image_preprocessing.py`, phase dataclasses in training scripts |
| Pretrain paths | `pretrain/phase2_simclr_retrain_top10_downstream.py`, `pretrain/phase3_simclr_full_pretrain.py` |
| Benchmark roster/report paths | `utils/config.py`, `test/phase7_test_benchmark.py`, `tools/export_torchscript_phase7_torch10.py` |
| Plot/report constants | `utils/config.py`, `preprocessing/dataset_figures.py`, `test/plot_phase7_*.py` |

### Reproducibility controls
Confirmed in `utils/repro.py` and used throughout the phase scripts:

| Control | Behavior |
| --- | --- |
| Global seed | default `42` |
| Deterministic worker seeding | `seed_worker()` |
| CuBLAS determinism env var | set in `set_global_seed()` |
| TF32 disabling in deterministic mode | explicit |
| `torch.use_deterministic_algorithms()` | enabled when deterministic |
| Seeded `torch.Generator` for DataLoaders | used across HPO and retrain scripts |
| Persisted file lists | Phase-1 and Phase-4 subset lists |

### Hardcoded path patterns
Confirmed examples:

| Path | Purpose |
| --- | --- |
| `../dataset/train`, `../dataset/valid`, `../dataset/test` | used by most train/eval scripts |
| `../pretrain/checkpoints/simclr_phase3/phase3_seed13_last.pth` | default transfer checkpoint |
| `bench_phase7/20260201_103832/...` and `bench_phase7_jetson_ts/20260308_103016/...` | plotting defaults in `utils/config.py` |
| `/work/tools/export_torchscript_10` | hardcoded in `tools/check_all_torchscript_models.py` |

## 10. Generated artifacts
Expected outputs confirmed by code:

| Stage | Artifacts |
| --- | --- |
| Preprocessing | `temp/*.png`, `dataset/{train,valid,test}/*.png`, `*_label.png` |
| Phase 1 | Optuna SQLite DB rows, persisted file lists in `runs/simclr_hpo/` |
| Phase 2 | `pretrain/checkpoints/simclr_topk_retrain_downstream/*.pth`, `phase2_topk_results.csv` |
| Phase 3 | `pretrain/checkpoints/simclr_phase3/*last.pth`, logs/metadata, optional diagnostics |
| Phase 4 | Optuna study rows in `UltraLightFCN_study.db`, fixed subset lists in `runs/hpo_subsets/` |
| Phase 5 | per-seed `last.pth`, `epoch_log.csv`, `phase5_topk_results.csv`, `phase5_winner.json` |
| Phase 6 | per-seed `last.pth`, `epoch_log.csv`, `phase6_test_per_seed.csv`, `phase6_test_report.json` |
| SOTA Stage 1 | per-seed `last.pth`, `epoch_log.csv`, `sota_stage1_results.csv`, `sota_stage1_aggregate.csv`, `stage1_winners.json` |
| SOTA Stage 2 | per-seed `last.pth`, `epoch_log.csv`, `phaseSOTA_seed_runs.csv`, `phaseSOTA_test_report.json` |
| Phase 7 desktop | `phase7_master_report.json`, timing CSVs, quality CSV, per-image `.npz` |
| TorchScript export | `.ts`, `.meta.json`, export index under `tools/export_torchscript_10/` |
| Jetson benchmark | per-model JSON/CSV reports under `bench_phase7_jetson_ts/<run>/` |
| Reporting utilities | plots in `bench_phase7/overall_plots`, `bench_phase7_jetson_ts/overall_plots`, `train/seg_phase5/ablation/`, `pretrain/aug_ablation/` |

## 11. Risks and technical debt
### Confirmed issues

| File | Issue | Why it matters |
| --- | --- | --- |
| `pretrain/phase3_simclr_full_pretrain.py:45` | Imports `from utils.metrics_simclr import ...`, but only `pretrain/utils/metrics_simclr.py` exists. | Phase-3 script will fail unless Python path is modified externally or a missing module is added. |
| `utils/config.py` | Plotting constants hardcode specific timestamped benchmark folders such as `bench_phase7/20260201_103832/...` and `bench_phase7_jetson_ts/20260308_103016/...`. | Plot scripts depend on a specific prior run layout and are fragile across fresh experiments. |
| `tools/check_all_torchscript_models.py` | Hardcodes `BASE_DIR = Path("/work/tools/export_torchscript_10")`. | Tool is environment-specific and will not work from the current Windows workspace without edits. |
| `test/smoke_ts.py` | Hardcodes `export_torchscript/...fullft...ts`, while the active exporter writes to `tools/export_torchscript_10/` and defaults to excluding `fullft`. | Smoke test is out of sync with the exporter and likely stale. |
| `requirements.txt` vs `tools/export_torchscript_phase7_torch10.py` | Main env pins `torch==2.7.1+cu128`, while the exporter explicitly targets Python 3.6 + Torch 1.10.x compatibility. | Deployment reproducibility depends on a second environment that is not fully encoded in the repo root requirements. |
| `models/UltraLightFCN_base.py` internal defaults vs `utils/config.py` | Base model defaults use `mini_aspp_gpool=False`, `sa_window_size=8`, `sa_dropout=0.0`, while experiment config uses `True`, `16`, `0.1`. | Any script instantiating `UltraLightFCN()` without passing `SEG_PARAMS` can silently create a different architecture. |
| `preprocessing/image_preprocessing.py` | The main split loop only handles `PV01`, `PV03`, `PV08`; `OTHER` is commented out. | New subsets will be silently excluded unless preprocessing is updated. |
| `utils/transforms.py:4` | Imports `from scipy.constants import value`, which is unused. | Minor hygiene issue; can confuse dependency intent. |

### Areas that are controlled but still fragile

| Area | Observation |
| --- | --- |
| Reproducibility | Seed management is strong, but most scripts run with `deterministic=False` for speed by default. |
| Data leakage | Group-aware split by `parent_id` prevents same-parent leakage after tiling, but cross-folder near-duplicates cannot be ruled out from code alone. |
| Training vs deployment parity | There is a dedicated deploy model and a TorchScript exporter, but multiple architecture default sources increase the chance of mismatch if metadata is incomplete. |
| Metric naming | The code uses both `soft_dice`, `hard_dice@0.5`, `val_soft`, `val_hard05`, `dice_hard05`; meaning is consistent but naming is not fully uniform. |

## 12. Questions for the project owner
1. Is the intended Phase-3 import path `pretrain.utils.metrics_simclr`, and if so, is the current `utils.metrics_simclr` import a known bug or an environment-specific path assumption?
2. Are the timestamped paths in `utils/config.py` meant to be edited manually per benchmark run, or is there an unstored convention for updating them automatically?
3. Is the Jetson deployment environment documented elsewhere? The root `requirements.txt` does not capture the Torch 1.10 / Python 3.6 constraint described in the exporter.
4. Should `OTHER` subset handling remain disabled permanently, or is it expected to be activated when new raw data appears?
5. Are there manuscript-facing definitions for why Phase-7 uses hard metrics only for reporting while Phases 4 and 5 select by soft Dice?

## 13. Recommended next steps
1. Document the two-environment story explicitly: main training environment vs TorchScript/Jetson export environment.
2. Consolidate architecture defaults so `UltraLightFCN` has one canonical parameter source.
3. Fix or validate the Phase-3 `metrics_simclr` import path before the next reproduction run.
4. Replace hardcoded benchmark/report timestamps with manifest discovery or CLI/config inputs.
5. Add a lightweight repository README that maps `Phase 1` through `Phase 7` to the exact scripts and expected artifacts.
6. Add a small static verification script that checks for required files, import paths, and phase artifact dependencies before running long experiments.

# Data Guide

## Raw Data Assumptions

`preprocessing/image_preprocessing.py` expects raw data under `../data` when launched from `preprocessing/`. The script recursively scans folders for BMP pairs in the same folder:

- Image: `<base>.bmp`
- Mask: `<base>_label.bmp`

Subset labels are inferred from filename prefixes matching `PVxx_`, for example `PV01`, `PV03`, and `PV08`. If a name does not match, the helper returns `OTHER`; the current main split loop processes subsets found in the sample set.

## Preprocessing Behavior

Phase 0 is implemented by `preprocessing/image_preprocessing.py`.

Key defaults:

| Setting | Value |
| --- | --- |
| Raw root | `../data` |
| Temp cache | `../temp` |
| Prepared dataset root | `../dataset` |
| Tile size | `256` |
| Stride | `256` |
| Positive coverage threshold | `0.005` |
| Minimum positive pixels | `64` |
| Negative quota multiplier | `alpha = 2.0` |
| Hard-negative fraction | `0.5` |
| Max negatives per positive parent | `4` |
| Empty-parent negative quota | `global_empty_alpha = 0.2`, cap `5000` |
| Split ratios | `(0.8, 0.1, 0.1)` |
| Seed | `42` |

The preprocessing flow:

1. Collect recursive BMP image/mask pairs.
2. Convert missing pairs to `../temp` PNGs with RGB images and binary masks in `{0,1}`.
3. Tile images at 256x256 with stride 256. Images already 256x256 produce a single tile.
4. Keep positive tiles by coverage or absolute positive-pixel threshold.
5. Select negatives by parent image, mixing texture-ranked hard negatives and random negatives.
6. Add limited negatives from parent images with zero positives.
7. Split group-aware by parent image within subsets.
8. Write split images and masks to `../dataset/<split>`.

The script has a guard that skips preprocessing if `train`, `valid`, and `test` each already contain at least one valid image/mask pair.

## Prepared Dataset Layout

Prepared split folders:

```text
dataset/
  train/
    <stem>.png
    <stem>_label.png
  valid/
    <stem>.png
    <stem>_label.png
  test/
    <stem>.png
    <stem>_label.png
```

`utils/dataset.py` expects masks next to images using `<stem>_label.png`. It excludes `*_label.png` from image lists and raises if a mask is missing.

## Current Discovered Dataset Counts

Lightweight file counting in this checkout found:

| Split | Images | Masks |
| --- | ---: | ---: |
| `dataset/train` | 22577 | 22577 |
| `dataset/valid` | 2995 | 2995 |
| `dataset/test` | 2825 | 2825 |

These counts were obtained without opening image contents or running preprocessing.

## Data Policy

`data/`, `dataset/`, and `temp/` are generated/local data folders, not source. They are ignored by the top-level `.gitignore`. Do not move, delete, or regenerate them unless explicitly requested.

## Troubleshooting

- Missing masks: `SolarPanelDataset` raises `Mask not found for image ... Expected <stem>_label.png`.
- Unexpected subset prefixes: preprocessing assigns non-`PVxx_` names to `OTHER`.
- Empty dataset: preprocessing and dataset classes raise if no valid images/pairs are found.
- Existing prepared dataset: Phase 0 skips work if all three prepared split folders already have valid pairs.

"""
STEP-BY-STEP image preprocessing with incremental TEMP for training a segmentation model.:
- Step 0/1: Validate the TEMP cache. Convert ONLY missing BMP pairs to PNG (mask -> {0,1}).
            If TEMP already contains ALL pairs, skip conversion entirely.
- Step 2:   Crop & select negatives per subset (PV01 / PV03 / PV08), keeping all positives.
- Step 3:   Stratified split BY subset (80/10/10), then write final files:
            IMAGES as .PNG (lossless) and MASKS as .PNG (lossless, 0/1).

Why this structure?
- Robust to partial reruns: if TEMP is already “complete”, we skip re-conversion.
- You can inspect TEMP to confirm normalization (e.g., masks are {0,1}) before cropping.
- Ensures each subset contributes to train/valid/test, keeping some empty (negative) tiles.

Dependencies: Pillow (PIL), NumPy, tqdm
"""

import os
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

# Pillow sometimes crashes on partially written/corrupt files (especially BMP).
# This flag tells Pillow to load what it can instead of crashing the whole script.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------
# CONFIG — tweak as needed
# -------------------------

# (1) Input and temporary/output directories
in_dir    = Path("data")        # Root of original BMPs (images + masks mixed in subfolders)
temp_dir  = Path("temp")        # Cache for normalized PNGs (images as RGB, masks as {0,1})
out_root  = Path("dataset")     # Final output (train/valid/test)

# (2) Tiling parameters
tile_size = 256
stride    = 256  # If you want overlap, set <256 (e.g., 128 for 50% overlap)

# (3) Positive/negative definition for tiles:
#     We mark a tile as positive if EITHER the relative coverage is high enough,
#     OR it has at least a minimum absolute number of positive pixels.
coverage_pos_thresh = 0.005  # >= 0.5% of pixels active => positive
min_pos_pixels      = 64     # OR at least 64 positive pixels => positive

# (4) Negative selection per parent image (only for parents with >=1 positive tile)
alpha               = 2.0   # quota multiplier; up to alpha * (#positives_from_that_parent) negatives
hard_neg_frac       = 0.5   # fraction of the quota taken from the “hardest” negatives (by texture)
max_neg_per_parent  = 4     # absolute cap of negatives per parent (after alpha)

# (5) Extra negatives from parents with zero positives (P==0) — done per subset
#     This ensures pure-negative parents don’t entirely disappear.
global_empty_alpha      = 0.2   # up to 20% of the subset's positive count
global_empty_cap        = 5000  # absolute safety limit per subset
global_empty_hard_frac  = 0.5   # within that portion, half hard, half random

# (6) Train/valid/test ratios (applied per subset)
split_ratios = (0.8, 0.1, 0.1)

# (7) Speed/quality knob for texture scoring: downscale tiles before gradient
#     128 is a good balance; 64 is faster; None means no downscale (slower).
texture_downscale: Optional[int] = 128

# (8) Seed for reproducibility (splits, random selections)
seed = 42

# -------------------------
# UTILS — general helpers
# -------------------------

def is_mask_name(name: str) -> bool:
    """Return True if filename looks like a mask by the '_label' suffix."""
    # We allow both BMP and PNG here because TEMP holds PNG masks.
    return name.lower().endswith("_label.bmp") or name.lower().endswith("_label.png")

def base_from_img_name(name: str) -> str:
    """
    Strip extension from image (BMP) filename.
    Example: 'PV01_314902_1196424.bmp' -> 'PV01_314902_1196424'
    """
    return name[:-4] if name.lower().endswith(".bmp") else name[:-4]

def base_from_mask_name(name: str) -> str:
    """
    Strip extension and '_label' suffix from mask name.
    Example: 'PV01_314902_1196424_label.bmp' -> 'PV01_314902_1196424'
    """
    stem = name[:-4]
    return stem.replace("_label", "")

def safe_outfile_base(s: str) -> str:
    """
    Convert a 'parent_id' (which includes relative path + base) into a flat, filesystem-safe base.
    We use this base to avoid collisions when writing to a single folder.
    Example: 'setA/day1/PV01_123' -> 'setA-day1-PV01_123'
    """
    s = s.replace('\\', '/').strip().strip('/')
    s = s.replace('/', '-')
    s = re.sub(r'[^A-Za-z0-9_\-.]', '_', s)
    return s

def get_subset_from_base(base: str) -> str:
    """
    Determine subset from base name, e.g., 'PV01_...' -> 'PV01'.
    If doesn't match the pattern, return 'OTHER'.
    """
    m = re.match(r"^(PV\d{2})_", base)
    return m.group(1) if m else "OTHER"

def ensure_dirs(*dirs: Path):
    """Create directories if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def load_image_rgb_np(path: Path) -> np.ndarray:
    """
    Load an image as HxWx3 uint8 (RGB).
    We read via Pillow, convert to RGB, and then cast to NumPy for fast ops.
    """
    with Image.open(path) as im:
        im.load()
        return np.asarray(im.convert("RGB"), dtype=np.uint8)

def load_mask_gray_np(path: Path) -> np.ndarray:
    """
    Load a mask as HxW uint8 (grayscale).
    Even if it's PNG in TEMP, we treat it as grayscale for uniformity.
    """
    with Image.open(path) as im:
        im.load()
        return np.asarray(im.convert("L"), dtype=np.uint8)

# -------------------------
# FINAL DATASET GUARD
# -------------------------

def count_pairs_in_split(split_dir: Path) -> int:
    """
    Count valid (image, mask) pairs in a split directory:
      image: <base>.png
      mask:  <base>_label.png
    Pair is valid if both files exist for the same <base>.
    """
    if not split_dir.exists():
        return 0
    # gather masks bases
    mask_bases = set()
    for p in split_dir.glob("*_label.png"):
        base = p.name[:-10]  # remove '_label.png' (10 chars)
        mask_bases.add(base)
    # count pairs where corresponding png exists
    pairs = 0
    for base in mask_bases:
        if (split_dir / f"{base}.png").exists():
            pairs += 1
    return pairs

def dataset_already_exists(out_root: Path) -> bool:
    """
    Return True if train/valid/test each contain at least one valid pair.
    (Lightweight guard; adjust if you require stricter completeness checks.)
    """
    train_pairs = count_pairs_in_split(out_root / "train")
    valid_pairs = count_pairs_in_split(out_root / "valid")
    test_pairs  = count_pairs_in_split(out_root / "test")
    if train_pairs > 0 and valid_pairs > 0 and test_pairs > 0:
        print(f"[GUARD] Existing dataset detected in '{out_root}'. "
              f"Pairs — train:{train_pairs}, valid:{valid_pairs}, test:{test_pairs}. Skipping preprocessing.")
        return True
    return False

# -------------------------
# STEP 0/1: BMP → TEMP PNG (mask {0,1}) — INCREMENTAL
# -------------------------

def collect_pairs_recursive_bmp(in_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Recursively scan the input tree and pair images and masks within each folder.
    Pairing rule: <base>.bmp ↔ <base>_label.bmp (same folder).
    Returns: list of (image_path, mask_path, parent_id)
      - parent_id encodes relative path + base to keep groups together and avoid name collisions
    """
    pairs = []
    for folder in [p for p in in_dir.rglob('*') if p.is_dir()]:
        bmp_files = list(folder.glob("*.bmp"))
        if not bmp_files:
            continue

        img_map: Dict[str, Path] = {}
        msk_map: Dict[str, Path] = {}

        for p in bmp_files:
            name = p.name
            if is_mask_name(name):
                msk_map[base_from_mask_name(name)] = p
            else:
                img_map[base_from_img_name(name)] = p

        rel_dir = folder.relative_to(in_dir)
        for base, ip in img_map.items():
            mp = msk_map.get(base)
            if mp is None:
                continue  # unmatched image — skip
            # parent_id includes relative directory + base
            parent_id = str(rel_dir / base) if str(rel_dir) != '.' else base
            pairs.append((ip, mp, parent_id))
    return pairs

def binarize_to_01(mask_arr: np.ndarray) -> np.ndarray:
    """
    Normalize mask to {0,1} (uint8).
    Threshold at 5: >=5 -> 1; else -> 0.
    Using 0/1 is nice for later metrics and smaller PNGs (often compress well).
    """
    return (mask_arr > 5).astype(np.uint8)

def temp_expected_names(parent_id: str) -> Tuple[str, str]:
    """
    Given a parent_id, compute the two expected TEMP filenames:
      - image: <safe_parent_id>.png
      - mask:  <safe_parent_id>_label.png
    """
    out_base = safe_outfile_base(parent_id)
    return f"{out_base}.png", f"{out_base}_label.png"

def index_temp_pairs(temp_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """
    Build an index from TEMP: base -> (image_png_path, mask_png_path).
    Here, 'base' is already the safe filename base (no directories).
    """
    img_map: Dict[str, Path] = {}
    msk_map: Dict[str, Path] = {}

    for p in temp_dir.glob("*.png"):
        name = p.name
        if name.lower().endswith("_label.png"):
            base = name[:-4].replace("_label", "")  # strip '.png' and '_label'
            msk_map[base] = p
        else:
            base = name[:-4]
            img_map[base] = p

    # Merge only pairs where both image and mask are present.
    pairs: Dict[str, Tuple[Path, Path]] = {}
    for base, ip in img_map.items():
        mp = msk_map.get(base)
        if mp is not None:
            pairs[base] = (ip, mp)
    return pairs

def step1_incremental_convert_to_temp(pairs_bmp: List[Tuple[Path, Path, str]], temp_dir: Path) -> None:
    """
    TEMP cache build/update:
    - Read what is already in TEMP.
    - Convert only the missing pairs from BMP -> PNG (mask normalized to {0,1}).
    - If nothing is missing, skip conversion entirely.

    Benefits:
    - Idempotent: safe to rerun without duplicating work.
    - You can inspect TEMP independently (sanity checks) before cropping/splitting.
    """
    ensure_dirs(temp_dir)

    # Index existing TEMP content
    existing = index_temp_pairs(temp_dir)

    # Determine which BMP pairs are missing in TEMP
    to_convert = []
    for img_path, mask_path, parent_id in pairs_bmp:
        base_img, _ = temp_expected_names(parent_id)  # expected image filename
        base = base_img[:-4]  # strip '.png' -> safe parent base
        if base not in existing:
            to_convert.append((img_path, mask_path, parent_id))

    if not to_convert:
        print("[STEP 1] TEMP is complete. Skipping conversion.")
        return

    print(f"[STEP 1] Converting {len(to_convert)} missing BMP pairs -> TEMP PNG (mask -> 0/1)...")
    bad = []
    for img_path, mask_path, parent_id in tqdm(to_convert):
        out_base = safe_outfile_base(parent_id)
        img_out = temp_dir / f"{out_base}.png"
        msk_out = temp_dir / f"{out_base}_label.png"
        try:
            # Load original BMPs
            img_np = load_image_rgb_np(img_path)
            msk_np = load_mask_gray_np(mask_path)
            # Normalize mask to 0/1
            msk_01 = binarize_to_01(msk_np)
            # Save to TEMP as PNG (lossless)
            Image.fromarray(img_np).convert("RGB").save(img_out, optimize=True)
            Image.fromarray(msk_01).convert("L").save(msk_out, optimize=True)
        except Exception as e:
            # If any file is corrupt/unreadable, log and continue
            bad.append((str(img_path), repr(e)))
            continue

    if bad:
        log = temp_dir / "bad_files_step1.txt"
        with open(log, "w", encoding="utf-8") as f:
            for p, err in bad:
                f.write(f"{p}\t{err}\n")
        print(f"[WARN] Step1 skipped {len(bad)} problematic files. See {log}")

# -------------------------
# STEP 2: CROP + NEGATIVE SELECTION (per subset)
# -------------------------

def texture_score_np(rgb_arr: np.ndarray, down_to: Optional[int] = 128) -> float:
    """
    A fast 'texture' proxy for negative mining:
      - Convert tile to grayscale (float32),
      - Optionally downscale to 'down_to' (e.g., 128x128) for speed,
      - Use np.gradient to compute gradient magnitude, then return mean magnitude.

    Rationale:
      - Hard negatives often have more structure/edges (rooftops, patterns).
      - np.gradient is vectorized C code (much faster than manual Sobel loops).
      - Downscaling preserves ranking fairly well and speeds up dramatically.
    """
    gray = rgb_arr.mean(axis=2).astype(np.float32)
    H, W = gray.shape
    if down_to is not None and (H != down_to or W != down_to):
        gray = np.asarray(
            Image.fromarray(gray.astype(np.uint8)).convert("L").resize((down_to, down_to), Image.BILINEAR),
            dtype=np.float32
        )
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    return float(mag.mean())

def crop_tiles_np(img_arr: np.ndarray, size=256, stride=256):
    """
    Generator yielding non-overlapping (or overlapping, if stride<size) crops.
    Uses NumPy slicing — no per-tile PIL .crop overhead.
    Yields: (r, c, tile_arr) where r,c are tile indices in the tiling grid.
    """
    H, W = img_arr.shape[:2]
    rows = (H - size) // stride + 1
    cols = (W - size) // stride + 1
    for r in range(rows):
        top = r * stride
        for c in range(cols):
            left = c * stride
            yield r, c, img_arr[top:top+size, left:left+size]

def is_positive_tile(mask01: np.ndarray) -> bool:
    """
    Decide whether a tile is positive based on:
      - relative coverage threshold (>= coverage_pos_thresh),
      - OR absolute count threshold (>= min_pos_pixels).
    This combination is robust for small objects.
    """
    cov = (mask01 > 0).mean()
    pos_pixels = int((mask01 > 0).sum())
    return (cov >= coverage_pos_thresh) or (pos_pixels >= min_pos_pixels)

def list_temp_pairs(temp_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Enumerate already converted pairs from TEMP.
    Returns (image_png_path, mask_png_path, parent_id_safe).
    Using the safe base as parent_id ensures stable grouping in later steps.
    """
    pairs = []
    existing = index_temp_pairs(temp_dir)  # base -> (img, msk)
    for base, (ip, mp) in existing.items():
        parent_id = base  # safe base acts as parent_id downstream
        pairs.append((ip, mp, parent_id))
    return pairs

def step2_crop_and_select(temp_pairs: List[Tuple[Path, Path, str]]):
    """
    This step produces the full pool of tiles to later split and write:
      - For 256x256 images: a single tile is produced (no crop).
      - For larger images: crop into (size=stride=256) tiles (no overlap).
      - Keep ALL positives.
      - For negatives:
          * Buffer per parent (for parents with P>0) and select up to a quota.
          * Additionally, collect negatives from parents with P==0 to a per-subset pool
            and later select a small portion ('global top-up') so pure-negative parents
            don't vanish entirely.
    Returns: List[Dict] of samples, each with fields:
      subset, parent_id, outfile_base, is_positive, img_np, mask_np({0,1}), texture
    """
    random.seed(seed)
    np.random.seed(seed)

    kept_positives: List[Dict] = []                       # all positive tiles
    neg_candidates_by_parent: Dict[str, List[Dict]] = {}  # negative tiles grouped by parent_id
    empty_parent_candidates_by_subset: Dict[str, List[Dict]] = {}  # negatives from P==0 parents per subset
    pos_count_by_parent: Dict[str, int] = {}              # number of positive tiles per parent

    print("[STEP 2] Cropping and buffering negatives (per subset)...")
    for img_png, mask_png, parent_id in tqdm(temp_pairs):
        # Derive subset label (PV01/PV03/PV08) from the last token in safe name,
        # which still begins with 'PVxx_'.
        base_for_subset = parent_id.split('-')[-1]
        subset = get_subset_from_base(base_for_subset)

        # Load the TEMP PNGs
        img_np = load_image_rgb_np(img_png)
        # TEMP masks are already {0,1}, but re-ensure strictly binary in case of anomalies
        msk01 = (load_mask_gray_np(mask_png) > 0).astype(np.uint8)

        H, W = img_np.shape[:2]
        has_positive = False  # track if this parent yields any positive tiles

        def add_sample(r_idx: int, c_idx: int, tile_img: np.ndarray, tile_mask: np.ndarray):
            """
            Inner helper to classify a tile, fill sample dict, and route it to the right buffer.
            """
            nonlocal has_positive
            pos = is_positive_tile(tile_mask)
            # outfile_base: if the parent was larger, append r/c; else keep parent base
            obase = f"{parent_id}_r{r_idx}_c{c_idx}" if (H != tile_size or W != tile_size) else parent_id

            sample = {
                "subset": subset,
                "parent_id": parent_id,
                "outfile_base": obase,      # used for writing final files
                "is_positive": bool(pos),
                "img_np": tile_img.copy(),  # ensure independent arrays (no views)
                "mask_np": tile_mask.copy(),
                "texture": 0.0,             # filled for negatives
            }

            if pos:
                kept_positives.append(sample)
                has_positive = True
            else:
                # Compute texture for negative mining (hard-negative prioritization).
                sample["texture"] = texture_score_np(tile_img, texture_downscale)
                neg_candidates_by_parent.setdefault(parent_id, []).append(sample)

        # Create tiles (or a single tile if exactly 256x256)
        if H >= tile_size and W >= tile_size:
            if H == tile_size and W == tile_size:
                # Single tile case: no cropping
                add_sample(0, 0, img_np, msk01)
            else:
                # Crop in a grid: cheap slicing via NumPy
                for r, c, tile in crop_tiles_np(img_np, size=tile_size, stride=stride):
                    tile_m = msk01[r*stride:r*stride+tile_size, c*stride:c*stride+tile_size]
                    add_sample(r, c, tile, tile_m)

        # Update parent positive count and/or buffer negatives from P==0 parents
        if has_positive:
            pos_count_by_parent[parent_id] = pos_count_by_parent.get(parent_id, 0) + 1
        else:
            # Parent has no positives at all → stash its negatives for the per-subset "global top-up"
            if parent_id in neg_candidates_by_parent and len(neg_candidates_by_parent[parent_id]) > 0:
                empty_parent_candidates_by_subset.setdefault(subset, []).extend(neg_candidates_by_parent[parent_id])

    # Now, select per-parent negatives for parents with P>0 according to the quota.
    print("[STEP 2] Selecting negatives per parent...")
    selected_negs: List[Dict] = []
    for parent_id, cands in neg_candidates_by_parent.items():
        P = pos_count_by_parent.get(parent_id, 0)  # how many positives from this parent
        if P <= 0:
            # Parents with P==0 handled in the 'global top-up' section.
            continue

        # Quota: minimum of alpha*P, the cap, and available candidates
        target = min(int(alpha * P), max_neg_per_parent, len(cands))
        if target <= 0:
            continue

        # Sort by texture desc → pick 'hard_neg_frac' hardest, rest random
        cands_sorted = sorted(cands, key=lambda s: s["texture"], reverse=True)
        k_hard = int(round(hard_neg_frac * target))
        hard_sel = cands_sorted[:k_hard]
        rest = cands_sorted[k_hard:]
        random.shuffle(rest)
        rand_sel = rest[:max(0, target - len(hard_sel))]
        selected_negs.extend(hard_sel + rand_sel)

    # Global top-up for parents with P==0, PER SUBSET (PV01/PV03/PV08):
    print("[STEP 2] Global top-up from empty parents per subset...")
    # Count positives per subset (for proportional quota)
    kept_by_subset_pos_count: Dict[str, int] = {}
    for s in kept_positives:
        kept_by_subset_pos_count[s["subset"]] = kept_by_subset_pos_count.get(s["subset"], 0) + 1

    selected_empty_negs: List[Dict] = []
    for subset, cands in empty_parent_candidates_by_subset.items():
        total_P_subset = kept_by_subset_pos_count.get(subset, 0)
        if total_P_subset == 0 or not cands:
            # If no positives at all in the subset, we skip to avoid flooding with easy negatives.
            continue

        # Quota for empty-parent negatives: proportional to #positives in this subset.
        target = min(int(global_empty_alpha * total_P_subset), global_empty_cap, len(cands))
        if target <= 0:
            continue

        # Again: hard portion + random remainder
        cands_sorted = sorted(cands, key=lambda s: s["texture"], reverse=True)
        k_hard = int(round(global_empty_hard_frac * target))
        hard_sel = cands_sorted[:k_hard]
        rest = cands_sorted[k_hard:]
        random.shuffle(rest)
        rand_sel = rest[:max(0, target - len(hard_sel))]
        selected_empty_negs.extend(hard_sel + rand_sel)

    # Merge final pool
    all_samples = kept_positives + selected_negs + selected_empty_negs
    return all_samples

# -------------------------
# STEP 3: SPLIT BY SUBSET & WRITE
# -------------------------

def split_by_parent_within_subset(samples: List[Dict], subset: str, ratios=(0.8, 0.1, 0.1)):
    """
    Perform group-aware splitting per subset:
      - group by parent_id (so all tiles from one original image stay together),
      - shuffle parents, then allocate to train/valid/test by the given ratios,
      - finally expand to sample lists.

    This prevents information leakage across splits.
    """
    random.seed(seed)

    # Collect indices of samples belonging to this subset
    idxs = [i for i, s in enumerate(samples) if s["subset"] == subset]

    # Group these indices by parent_id
    by_parent: Dict[str, List[int]] = {}
    for i in idxs:
        by_parent.setdefault(samples[i]["parent_id"], []).append(i)

    parents = list(by_parent.keys())
    random.shuffle(parents)

    n = len(parents)
    if n == 0:
        return [], [], []

    n_train = int(round(ratios[0] * n))
    n_valid = int(round(ratios[1] * n))
    n_test  = n - n_train - n_valid

    train_par = set(parents[:n_train])
    valid_par = set(parents[n_train:n_train+n_valid])
    test_par  = set(parents[n_train+n_valid:])

    # Expand parent groups into sample indices
    train_idx, valid_idx, test_idx = [], [], []
    for p, arr in by_parent.items():
        if p in train_par: train_idx.extend(arr)
        elif p in valid_par: valid_idx.extend(arr)
        else: test_idx.extend(arr)

    return [samples[i] for i in train_idx], [samples[i] for i in valid_idx], [samples[i] for i in test_idx]

def ensure_some_negatives_each_subset(split_samples: List[Dict], subset: str, split_name: str):
    """
    Optional diagnostic: warn if a split has zero negatives for a subset that exists in that split.
    This is best-effort; you can increase alpha/global_empty_alpha or change the seed if needed.
    """
    sub = [s for s in split_samples if s["subset"] == subset]
    if not sub:
        return
    neg = sum(1 for s in sub if not s["is_positive"])
    if neg == 0:
        print(f"[WARN] {split_name}: subset {subset} has ZERO negatives. "
              f"Consider increasing alpha/global_empty_alpha or changing seed.")

def write_split(split_samples: List[Dict], split_name: str):
    """
    FINAL WRITE (requested change):
      - IMAGE saved as PNG (.png) (lossless) to avoid compression artifacts near edges.
      - MASK saved as PNG (.png), with {0,1} values (uint8). PNG is lossless,
        which is critical for segmentation labels.
      - Both files go into the SAME folder: out_root/<split>/
      - Filenames are aligned via the shared 'outfile_base':
           image: <outfile_base>.png
           mask:  <outfile_base>_label.png
    """
    out_dir = out_root / split_name
    ensure_dirs(out_dir)

    for s in tqdm(split_samples, desc=f"Writing {split_name}"):
        img_out = out_dir / f"{s['outfile_base']}.png"         # <-- image as PNG
        msk_out = out_dir / f"{s['outfile_base']}_label.png"   # <-- mask as PNG

        # Save image as PNG — lossless. This avoids compression artifacts near panel boundaries.
        Image.fromarray(s["img_np"]).convert("RGB").save(
            img_out,
            format="PNG",
            optimize=True
        )

        # Save mask as PNG — lossless, preserving exact 0/1 values
        Image.fromarray(s["mask_np"].astype(np.uint8)).convert("L").save(msk_out, optimize=True)

def report(name: str, lst: List[Dict]):
    """Simple split summary for quick sanity checking of class balance."""
    P = sum(1 for s in lst if s["is_positive"])
    N = sum(1 for s in lst if not s["is_positive"])
    print(f"{name:>5}: total={len(lst):6d}  pos={P:6d}  neg={N:6d}  pos_ratio={(P/(len(lst)+1e-9)):.3f}")

# -------------------------
# MAIN — orchestrates the 3 steps
# -------------------------

def main():
    random.seed(seed)
    np.random.seed(seed)

    # If final dataset already exists, skip everything
    if dataset_already_exists(out_root):
        return

    # Ensure output structure exists
    ensure_dirs(temp_dir, out_root / "train", out_root / "valid", out_root / "test")

    # --- STEP 0/1: Collect BMP pairs and update TEMP incrementally ---
    bmp_pairs = collect_pairs_recursive_bmp(in_dir)
    if not bmp_pairs:
        print("[FATAL] No BMP image/mask pairs found in input.")
        return

    # Convert ONLY missing pairs into TEMP (mask normalized to {0,1})
    step1_incremental_convert_to_temp(bmp_pairs, temp_dir)

    # From this point forward, we proceed exclusively from TEMP (PNG cache).
    temp_pairs = list_temp_pairs(temp_dir)
    if not temp_pairs:
        print("[FATAL] No PNG pairs found in TEMP after Step 1.")
        return

    # --- STEP 2: Crop + per-subset selection ---
    all_samples = step2_crop_and_select(temp_pairs)
    if not all_samples:
        print("[FATAL] No samples after cropping/selection.")
        return

    # --- STEP 3: Per-subset split, then merge and write ---
    subsets = ["PV01", "PV03", "PV08"]  # explicitly split these; others can be added if needed

    train_all: List[Dict] = []
    valid_all: List[Dict] = []
    test_all:  List[Dict] = []

    print("[STEP 3] Stratified split per subset (PV01/PV03/PV08)...")
    for sub in subsets:
        tr, va, te = split_by_parent_within_subset(all_samples, sub, ratios=split_ratios)
        train_all.extend(tr); valid_all.extend(va); test_all.extend(te)

    # (Optional) If you also want to handle “OTHER” subset, uncomment:
    # tr_o, va_o, te_o = split_by_parent_within_subset(all_samples, "OTHER", ratios=split_ratios)
    # train_all.extend(tr_o); valid_all.extend(va_o); test_all.extend(te_o)

    # Diagnostics: ensure each subset has some negatives in each split (best-effort)
    for sub in subsets:
        ensure_some_negatives_each_subset(train_all, sub, "train")
        ensure_some_negatives_each_subset(valid_all, sub, "valid")
        ensure_some_negatives_each_subset(test_all,  sub, "test")

    # FINAL WRITE — IMAGES = .png, MASKS = .png
    write_split(train_all, "train")
    write_split(valid_all, "valid")
    write_split(test_all,  "test")

    # Split-level summaries
    print("\n=== FINAL REPORT ===")
    report("train", train_all)
    report("valid", valid_all)
    report("test",  test_all)

if __name__ == "__main__":
    main()

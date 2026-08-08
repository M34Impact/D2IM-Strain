"""
Summary statistics of the relative strain error, per slice.

Addresses Reviewer 1, Comment 3: provides a quantitative comparison of the
displacement-derived (D2IM) and directly predicted (D2IM-Strain) strain fields
for the two cases shown in Figure 3.

Run from anywhere:
    python Review/figure3_error_stats.py
    python Review/figure3_error_stats.py --cases S5_INT_UL_ML_50_0007 S8_LES_UL_ML_50_0010

Notes on why this does not simply reuse the Main.py pipeline:

1. The three data folders are not the same size (Scan 251, Mask 419, W 413).
   Main.py pairs them by list position, which only works if all three folders
   hold exactly the same files. Here the Scan folder is a strict subset of the
   other two, so this script pairs them by filename instead and keeps the 251
   samples common to all three.

2. FolderImageLoader globs without sorting, so list order is not reproducible
   across machines. Because the train/val/test split is a shuffle of that list,
   the published split cannot be reconstructed reliably. Statistics for a named
   slice do not depend on the split, so this script reports per-slice results
   keyed by filename and writes the full table to CSV.
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

# The project modules and the data folders are addressed relative to the
# repository root, so resolve it here and work from there regardless of the
# directory the script was invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom

# The project code imports `tiffile`, which is a shim for `tifffile`; only the
# latter is pinned in requirements.txt, so accept whichever is installed.
try:
    import tifffile as tiff
except ImportError:
    import tiffile as tiff

from DataProcessing.Masking import Masking
from DataProcessing.Strain import Strain
from Training.DataSpilt import DataSplit

SCAN_DIR = PROJECT_ROOT / "Data" / "Input" / "Scan"
MASK_DIR = PROJECT_ROOT / "Data" / "Input" / "Mask"
W_DIR = PROJECT_ROOT / "Data" / "Target" / "W"

DISPLACEMENT_MODEL = PROJECT_ROOT / "Main" / "D2IM_trained.h5"
DIRECT_MODEL = PROJECT_ROOT / "Main" / "M1_best.h5"

CSV_OUT = PROJECT_ROOT / "Review" / "figure3_error_stats.csv"

NODE_SPACING = 50  # Node spacing, as used in Strain and DisplacementModel

# Test-set positions plotted as Figure 3, from `plot_num` in
# TrainingAnalysis.visualise_strain(). Resolved to filenames below.
FIG3_TEST_INDICES = [9, 3]  # 9 = Fig 3A (intact), 3 = Fig 3B (lesioned)


def legacy_order_cases():
    """
    Resolve the Figure 3 test indices to filenames using the file order that
    produced the published figures.

    FolderImageLoader collects files by globbing each supported extension in
    turn and never sorts, so the order is whatever the filesystem returns.
    That is not portable between machines, but it is reproducible on the
    machine the figures were generated on, which is what matters here.
    """
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(str(SCAN_DIR), f'*{ext}')))
    stems = [Path(f).stem for f in files]

    _, _, test = DataSplit(stems).split_data()
    return [test[i] if i < len(test) else None for i in FIG3_TEST_INDICES]


def aligned_stems():
    """Filenames (without extension) present in all three folders, ordered."""
    stems = [{p.stem for p in d.glob("*.tif")} for d in (SCAN_DIR, MASK_DIR, W_DIR)]
    common = stems[0] & stems[1] & stems[2]
    # Plain sort: results are keyed by filename, so only reproducibility matters.
    return sorted(common)


def load_stack(folder, stems, size):
    """Load and resize a stack of slices, matching LoadingData.ImageLoader."""
    images = []
    for stem in stems:
        img = tiff.imread(str(folder / f"{stem}.tif"))
        img[np.isnan(img)] = 0
        images.append(zoom(img, (size / img.shape[0], size / img.shape[1]),
                           mode="nearest", order=0))
    return np.array(images)


def relative_error(target, predicted):
    """Relative strain error in %, matching TrainingAnalysis.visualise_strain."""
    error = np.abs((target - predicted) / target) * 100
    return np.nan_to_num(error, posinf=0, neginf=0)


def summarise(error, valid):
    """Summary statistics over the valid (bone, non-zero target) cells only."""
    vals = error[valid]
    if vals.size == 0:
        return {k: float("nan") for k in ("n", "mean", "sd", "median", "p95", "max")}
    return {
        "n": vals.size,
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }


def print_table(title, rows):
    print(f"\n{title}")
    print("-" * 78)
    print(f"{'case':<26}{'n':>6}{'mean':>10}{'SD':>10}{'median':>10}{'p95':>10}{'max':>10}")
    print("-" * 78)
    for label, s in rows:
        print(f"{label:<26}{s['n']:>6}{s['mean']:>10.1f}{s['sd']:>10.1f}"
              f"{s['median']:>10.1f}{s['p95']:>10.1f}{s['max']:>10.1f}")
    print("-" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs=2, default=None,
                        metavar=("FIG3A", "FIG3B"),
                        help="filenames (without .tif) of the two Figure 3 cases; "
                             "defaults to resolving them from the original file order")
    args = parser.parse_args()

    cases = args.cases or legacy_order_cases()
    source = "given on the command line" if args.cases else \
             "resolved from the original file order (test indices 9 and 3)"
    print(f"Figure 3 cases {source}:")
    for label, stem in zip(("Fig 3A", "Fig 3B"), cases):
        print(f"  {label}: {stem}")

    # ---- Build a filename-aligned dataset ------------------------------------
    stems = aligned_stems()
    print(f"Scan {len(list(SCAN_DIR.glob('*.tif')))}, "
          f"Mask {len(list(MASK_DIR.glob('*.tif')))}, "
          f"W {len(list(W_DIR.glob('*.tif')))} files; "
          f"{len(stems)} common to all three and used here.")

    scan_array = load_stack(SCAN_DIR, stems, 256) / 255
    mask_array = load_stack(MASK_DIR, stems, 20)
    w_array = load_stack(W_DIR, stems, 20)

    mask_obj = Masking(mask_array)
    be_mask = mask_obj.get_binary_erosion_mask()
    bd_mask = mask_obj.get_binary_dilation_mask()

    # ---- Ground-truth strain in microstrain ----------------------------------
    strain_obj = Strain(be_mask, w_array)
    global_mean_S, global_std_S = strain_obj.global_mean, strain_obj.global_std

    # ---- Displacement-derived strain (D2IM) ----------------------------------
    displacement_model = tf.keras.models.load_model(DISPLACEMENT_MODEL)
    displacement_predictions = displacement_model.predict([scan_array, bd_mask])

    derived_ezz = []
    for i in range(scan_array.shape[0]):
        w = displacement_predictions[2][i].reshape(20, 20)
        derived_ezz.append(np.gradient(w, NODE_SPACING, axis=0) * 1e6)
    derived_ezz = np.where(be_mask, np.array(derived_ezz), 0.0)

    # DisplacementModel standardises the derived field by its OWN mean/std,
    # but TrainingAnalysis de-standardises it with the target's mean/std.
    # Both variants are reported so the effect of that mismatch can be judged.
    global_mean_D, global_std_D = np.mean(derived_ezz), np.std(derived_ezz)
    standardized_pezz = np.where(be_mask, (derived_ezz - global_mean_D) / global_std_D, 0.0)

    print(f"\ntarget  strain: mean={global_mean_S:.2f}, std={global_std_S:.2f}")
    print(f"derived strain: mean={global_mean_D:.2f}, std={global_std_D:.2f}")
    print(f"scale factor applied by the as-published path: {global_std_S / global_std_D:.4f}")

    # ---- Direct strain prediction (D2IM-Strain) ------------------------------
    direct_model = tf.keras.models.load_model(DIRECT_MODEL)
    n = scan_array.shape[0]
    input_scan = scan_array.reshape(n, scan_array.shape[1], scan_array.shape[2], 1)
    input_mask = be_mask.reshape(n, be_mask.shape[1], be_mask.shape[2], 1)
    direct_predictions = direct_model.predict([input_scan, input_mask])

    # ---- Fields in microstrain, flattened to one row per slice ---------------
    mask_flat = be_mask.reshape(n, 400)
    target = np.where(mask_flat, strain_obj.standardized_ezz.reshape(n, 400)
                      * global_std_S + global_mean_S, 0.0)
    direct = np.where(mask_flat, direct_predictions * global_std_S + global_mean_S, 0.0)
    derived_as_published = np.where(
        mask_flat, standardized_pezz.reshape(n, 400) * global_std_S + global_mean_S, 0.0)
    derived_corrected = np.where(mask_flat, derived_ezz.reshape(n, 400), 0.0)

    # ---- Which slices land in the test split under this ordering -------------
    # Indicative only: the published split used an unsorted glob order that
    # cannot be reproduced, so treat this as a cross-check, not ground truth.
    _, _, test_stems = DataSplit(list(stems)).split_data()
    test_set = set(test_stems)

    # ---- Per-slice statistics, written to CSV --------------------------------
    variants = {
        "derived_as_published": derived_as_published,
        "derived_corrected": derived_corrected,
        "direct": direct,
    }
    metrics = ("n", "mean", "sd", "median", "p95", "max")

    with open(CSV_OUT, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["filename", "specimen", "lesion", "orientation", "in_test_split"]
        for name in variants:
            header += [f"{name}_{m}" for m in metrics]
        writer.writerow(header)

        per_slice = {}
        for i, stem in enumerate(stems):
            parts = stem.split("_")
            t = target[i]
            valid = mask_flat[i].astype(bool) & (t != 0)

            stats = {name: summarise(relative_error(t, field[i]), valid)
                     for name, field in variants.items()}
            per_slice[stem] = stats

            row = [stem, parts[0], parts[1], parts[3], stem in test_set]
            for name in variants:
                row += [stats[name][m] for m in metrics]
            writer.writerow(row)

    print(f"\nPer-slice statistics for all {n} slices written to {CSV_OUT}")

    # ---- The two Figure 3 cases ----------------------------------------------
    for label, stem in zip(("Fig 3A", "Fig 3B"), cases):
        print(f"\n\n{'=' * 78}")
        if stem not in per_slice:
            print(f"{label}: '{stem}' not found. Pick from the CSV, or pass --cases.")
            print("=" * 78)
            continue

        lesion = "LESION" if "_LES_" in stem else "intact"
        where = "in test split" if stem in test_set else "NOT in test split"
        print(f"{label}  |  {stem}  ({lesion}, {where})")
        print("=" * 78)
        print_table("Relative strain error (%), bone cells only",
                    [("derived (as published)", per_slice[stem]["derived_as_published"]),
                     ("derived (corrected)", per_slice[stem]["derived_corrected"]),
                     ("direct (D2IM-Strain)", per_slice[stem]["direct"])])

"""
Specimen-level comparison of the displacement-derived and direct strain errors.

Addresses Reviewer 1, Comment 6: the Mann-Whitney test reported for Figure 2B
treats each DVC window as an independent observation. Windows within a specimen
are spatially correlated, so the effective sample size is inflated and the
p-value is optimistic. This script instead treats the specimen as the
experimental unit: one summary error per specimen per method, compared with a
paired test.

Run from anywhere:
    python Review/specimen_level_stats.py

Only test-set slices are used, so the comparison remains a held-out one. The
split is reproduced using the same unsorted glob order as FolderImageLoader,
which is what generated the published figures.
"""

import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
from scipy.stats import wilcoxon, binomtest

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

NODE_SPACING = 50
YIELD_THRESHOLD = 10000  # microstrain, as used throughout the manuscript


def aligned_stems():
    """Filenames common to all three folders. The folders differ in size."""
    sets = [{p.stem for p in d.glob("*.tif")} for d in (SCAN_DIR, MASK_DIR, W_DIR)]
    return sorted(sets[0] & sets[1] & sets[2])


def legacy_test_stems():
    """Test-set filenames under FolderImageLoader's unsorted glob order."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(str(SCAN_DIR), f'*{ext}')))
    _, _, test = DataSplit([Path(f).stem for f in files]).split_data()
    return list(test)


def load_stack(folder, stems, size):
    images = []
    for stem in stems:
        img = tiff.imread(str(folder / f"{stem}.tif"))
        img[np.isnan(img)] = 0
        images.append(zoom(img, (size / img.shape[0], size / img.shape[1]),
                           mode="nearest", order=0))
    return np.array(images)


def relative_error(target, predicted):
    err = np.abs((target - predicted) / target) * 100
    return np.nan_to_num(err, posinf=0, neginf=0)


def paired_report(label, specimens, derived, direct):
    """Paired test across specimens, with the sign test as a small-n backstop."""
    d = np.array(derived) - np.array(direct)
    n = len(d)
    better = int(np.sum(d > 0))
    print(f"\n{label}")
    print("-" * 74)
    print(f"{'specimen':<14}{'derived':>12}{'direct':>12}{'difference':>14}")
    print("-" * 74)
    for s, a, b in zip(specimens, derived, direct):
        print(f"{s:<14}{a:>12.1f}{b:>12.1f}{a - b:>14.1f}")
    print("-" * 74)
    print(f"n specimens = {n};  direct lower in {better}/{n}")
    if n >= 3 and np.any(d != 0):
        stat, p = wilcoxon(derived, direct)
        print(f"Wilcoxon signed-rank: W = {stat:.1f}, p = {p:.4f}")
    else:
        print("Wilcoxon signed-rank: not computable at this sample size")
    print(f"Sign test: p = {binomtest(better, n).pvalue:.4f}")


if __name__ == "__main__":
    stems = aligned_stems()
    test_stems = [s for s in legacy_test_stems() if s in set(stems)]
    print(f"{len(stems)} aligned slices; {len(test_stems)} in the test split")

    scan = load_stack(SCAN_DIR, stems, 256) / 255
    masks = load_stack(MASK_DIR, stems, 20)
    w = load_stack(W_DIR, stems, 20)

    mask_obj = Masking(masks)
    be_mask, bd_mask = mask_obj.get_binary_erosion_mask(), mask_obj.get_binary_dilation_mask()

    strain_obj = Strain(be_mask, w)
    mean_S, std_S = strain_obj.global_mean, strain_obj.global_std

    # Displacement-derived strain, kept in microstrain (see Review notes on the
    # standardisation mismatch in DisplacementModel/TrainingAnalysis).
    disp = tf.keras.models.load_model(DISPLACEMENT_MODEL).predict([scan, bd_mask])
    derived = np.array([np.gradient(disp[2][i].reshape(20, 20), NODE_SPACING, axis=0) * 1e6
                        for i in range(scan.shape[0])])
    derived = np.where(be_mask, derived, 0.0)

    direct_pred = tf.keras.models.load_model(DIRECT_MODEL).predict(
        [scan.reshape(-1, scan.shape[1], scan.shape[2], 1),
         be_mask.reshape(-1, 20, 20, 1)])

    n = len(stems)
    mask_flat = be_mask.reshape(n, 400)
    target = np.where(mask_flat, strain_obj.standardized_ezz.reshape(n, 400) * std_S + mean_S, 0.0)
    direct = np.where(mask_flat, direct_pred * std_S + mean_S, 0.0)
    derived = np.where(mask_flat, derived.reshape(n, 400), 0.0)

    index = {s: i for i, s in enumerate(stems)}

    # Group the test slices by specimen, pooling all their bone windows.
    by_specimen = {}
    for s in test_stems:
        by_specimen.setdefault(s.split("_")[0], []).append(index[s])

    regimes = {
        "All bone windows": lambda t: np.ones_like(t, dtype=bool),
        f"Below yield (|ezz| < {YIELD_THRESHOLD} ue)": lambda t: np.abs(t) < YIELD_THRESHOLD,
        f"At or above yield (|ezz| >= {YIELD_THRESHOLD} ue)": lambda t: np.abs(t) >= YIELD_THRESHOLD,
    }

    for label, regime in regimes.items():
        specimens, dv, dr = [], [], []
        for spec in sorted(by_specimen, key=lambda x: int(x[1:])):
            e_dv, e_dr = [], []
            for i in by_specimen[spec]:
                t = target[i]
                valid = mask_flat[i].astype(bool) & (t != 0) & regime(t)
                if not valid.any():
                    continue
                e_dv.append(relative_error(t, derived[i])[valid])
                e_dr.append(relative_error(t, direct[i])[valid])
            if not e_dv:
                continue
            specimens.append(spec)
            dv.append(float(np.median(np.concatenate(e_dv))))
            dr.append(float(np.median(np.concatenate(e_dr))))
        if len(specimens) >= 2:
            paired_report(f"{label}: median relative error (%) per specimen",
                          specimens, dv, dr)
        else:
            print(f"\n{label}: too few specimens with data in this regime")

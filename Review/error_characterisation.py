"""
Where the predictions are reliable: error against strain magnitude.

Addresses Reviewer 1, Comment 10, which asks for a more comprehensive analysis of
the error distribution so that reliable and unreliable regions can be told apart.

Run from anywhere, on the machine that generated the published figures:
    python Review/error_characterisation.py

Three blocks:

  (a) Error binned by measured strain magnitude. Reports median relative error
      and median absolute error per bin. Relative error diverges where the
      measured strain approaches zero, so the two together separate a genuine
      loss of accuracy from an artefact of a small denominator. Intended as
      Table 2 of the manuscript.

  (b) High-strain classification per specimen, labelled intact or lesioned, to
      show whether reliability is consistent across specimens and whether
      lesioned cases behave differently.

  (c) Where the misclassified windows sit. If missed high-strain windows cluster
      just above the threshold rather than at peak strain, the predictions are
      dependable where it matters most.
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
YIELD_THRESHOLD = 10000

# Upper edges, in microstrain. The last bin is everything at or above yield.
BINS = [1000, 2500, 5000, YIELD_THRESHOLD, np.inf]


def aligned_stems():
    sets = [{p.stem for p in d.glob("*.tif")} for d in (SCAN_DIR, MASK_DIR, W_DIR)]
    return sorted(sets[0] & sets[1] & sets[2])


def legacy_test_stems():
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(str(SCAN_DIR), f'*{ext}')))
    _, _, test = DataSplit([Path(f).stem for f in files]).split_data()
    return list(test)


def load_stack(folder, stems, size):
    out = []
    for stem in stems:
        img = tiff.imread(str(folder / f"{stem}.tif"))
        img[np.isnan(img)] = 0
        out.append(zoom(img, (size / img.shape[0], size / img.shape[1]),
                        mode="nearest", order=0))
    return np.array(out)


def relative_error(target, predicted):
    err = np.abs((target - predicted) / target) * 100
    return np.nan_to_num(err, posinf=0, neginf=0)


def classification(target, predicted):
    actual = np.abs(target) >= YIELD_THRESHOLD
    flagged = np.abs(predicted) >= YIELD_THRESHOLD
    tp = int(np.sum(actual & flagged)); fp = int(np.sum(~actual & flagged))
    fn = int(np.sum(actual & ~flagged)); tn = int(np.sum(~actual & ~flagged))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else float("nan")
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sens": sens, "spec": spec, "prec": prec, "f1": f1}


def bin_label(lo, hi):
    if lo == 0:
        return f"< {int(hi)}"
    return f"{int(lo)} to {int(hi)}" if np.isfinite(hi) else f">= {int(lo)}"


if __name__ == "__main__":
    stems = aligned_stems()
    test_stems = legacy_test_stems()
    print(f"{len(stems)} aligned slices, {len(test_stems)} in the test split\n")

    scan = load_stack(SCAN_DIR, stems, 256) / 255
    masks = load_stack(MASK_DIR, stems, 20)
    w = load_stack(W_DIR, stems, 20)

    mask_obj = Masking(masks)
    be_mask = mask_obj.get_binary_erosion_mask()
    bd_mask = mask_obj.get_binary_dilation_mask()

    strain_obj = Strain(be_mask, w)
    mean_S, std_S = strain_obj.global_mean, strain_obj.global_std

    disp = tf.keras.models.load_model(DISPLACEMENT_MODEL).predict([scan, bd_mask])
    derived_raw = np.array([np.gradient(disp[2][i].reshape(20, 20), NODE_SPACING, axis=0) * 1e6
                            for i in range(scan.shape[0])])
    derived_raw = np.where(be_mask, derived_raw, 0.0)
    mean_D, std_D = np.mean(derived_raw), np.std(derived_raw)
    standardized_pezz = np.where(be_mask, (derived_raw - mean_D) / std_D, 0.0)

    direct_pred = tf.keras.models.load_model(DIRECT_MODEL).predict(
        [scan.reshape(-1, scan.shape[1], scan.shape[2], 1),
         be_mask.reshape(-1, 20, 20, 1)])

    n = len(stems)
    mask_flat = be_mask.reshape(n, 400)
    target = np.where(mask_flat, strain_obj.standardized_ezz.reshape(n, 400) * std_S + mean_S, 0.0)
    fields = {
        "derived (as published)": np.where(
            mask_flat, standardized_pezz.reshape(n, 400) * std_S + mean_S, 0.0),
        "derived (corrected)": np.where(mask_flat, derived_raw.reshape(n, 400), 0.0),
        "direct (D2IM-Strain)": np.where(mask_flat, direct_pred * std_S + mean_S, 0.0),
    }

    index = {s: i for i, s in enumerate(stems)}
    test_idx = [index[s] for s in test_stems if s in index]

    # Pool every valid bone window across the test slices.
    valid = mask_flat[test_idx].astype(bool) & (target[test_idx] != 0)
    t_pool = target[test_idx][valid]
    pooled = {k: v[test_idx][valid] for k, v in fields.items()}

    # ---- (a) error against strain magnitude ---------------------------------
    print("=" * 92)
    print("(a) Error by measured strain magnitude, test-set bone windows")
    print("=" * 92)
    header = f"{'|measured ezz| (ue)':<22}{'n':>7}"
    for k in fields:
        header += f"{k.split(' ')[0][:8] + ' rel%':>16}"
    print(header)
    print("-" * 92)
    lo = 0
    rows = []
    for hi in BINS:
        sel = (np.abs(t_pool) >= lo) & (np.abs(t_pool) < hi)
        if sel.sum() == 0:
            lo = hi
            continue
        rel = {k: float(np.median(relative_error(t_pool[sel], v[sel]))) for k, v in pooled.items()}
        absl = {k: float(np.median(np.abs(t_pool[sel] - v[sel]))) for k, v in pooled.items()}
        rows.append((bin_label(lo, hi), int(sel.sum()), rel, absl))
        lo = hi
    for label, count, rel, _ in rows:
        line = f"{label:<22}{count:>7}"
        for k in fields:
            line += f"{rel[k]:>16.1f}"
        print(line)
    print("-" * 92)
    print("\nSame bins, median ABSOLUTE error (ue)")
    print("-" * 92)
    for label, count, _, absl in rows:
        line = f"{label:<22}{count:>7}"
        for k in fields:
            line += f"{absl[k]:>16.0f}"
        print(line)
    print("-" * 92)

    # ---- (b) classification per specimen ------------------------------------
    print("\n" + "=" * 92)
    print("(b) High-strain classification per specimen")
    print("=" * 92)
    groups = {}
    for s in test_stems:
        if s in index:
            groups.setdefault(s.split("_")[0], []).append(index[s])
    print(f"{'specimen':<10}{'type':<9}{'windows':>9}"
          f"{'derived sens':>14}{'direct sens':>13}"
          f"{'derived spec':>14}{'direct spec':>13}"
          f"{'derived F1':>12}{'direct F1':>11}")
    print("-" * 92)
    for spec in sorted(groups, key=lambda k: int(k[1:])):
        idx = groups[spec]
        v = mask_flat[idx].astype(bool) & (target[idx] != 0)
        t = target[idx][v]
        if t.size == 0:
            continue
        dv = classification(t, fields["derived (as published)"][idx][v])
        dr = classification(t, fields["direct (D2IM-Strain)"][idx][v])
        kind = "lesioned" if any("_LES_" in s for s in test_stems
                                 if s.split("_")[0] == spec) else "intact"
        print(f"{spec:<10}{kind:<9}{t.size:>9}"
              f"{dv['sens']:>14.3f}{dr['sens']:>13.3f}"
              f"{dv['spec']:>14.3f}{dr['spec']:>13.3f}"
              f"{dv['f1']:>12.3f}{dr['f1']:>11.3f}")
    print("-" * 92)

    # ---- (c) where the misclassified windows sit ----------------------------
    print("\n" + "=" * 92)
    print("(c) Measured strain magnitude of the misclassified windows")
    print("=" * 92)
    print(f"{'field':<24}{'n FN':>7}{'FN median':>12}{'FN < 1.5x thr':>15}"
          f"{'n FP':>7}{'FP median':>12}{'FP > 0.5x thr':>15}")
    print("-" * 92)
    for k, v in pooled.items():
        actual = np.abs(t_pool) >= YIELD_THRESHOLD
        flagged = np.abs(v) >= YIELD_THRESHOLD
        fn = np.abs(t_pool[actual & ~flagged])
        fp = np.abs(t_pool[~actual & flagged])
        fn_near = float(np.mean(fn < 1.5 * YIELD_THRESHOLD) * 100) if fn.size else float("nan")
        fp_near = float(np.mean(fp > 0.5 * YIELD_THRESHOLD) * 100) if fp.size else float("nan")
        print(f"{k:<24}{fn.size:>7}{np.median(fn) if fn.size else float('nan'):>12.0f}"
              f"{fn_near:>14.1f}%"
              f"{fp.size:>7}{np.median(fp) if fp.size else float('nan'):>12.0f}"
              f"{fp_near:>14.1f}%")
    print("-" * 92)
    print("\nFN = high-strain window predicted below threshold; "
          "FP = low-strain window predicted above it.")
    print(f"Threshold = {YIELD_THRESHOLD} ue. 'FN < 1.5x thr' is the share of missed "
          f"windows whose measured strain is below {int(1.5 * YIELD_THRESHOLD)} ue,")
    print("that is, close to the threshold rather than at peak strain.")

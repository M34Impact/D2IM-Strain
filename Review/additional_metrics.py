"""
Additional performance metrics, and the relative error close to the lesion.

Addresses Reviewer 1, Comment 8 (metrics beyond R2, and the claimed ~90% error
around lesions) and Reviewer 2, Comment 12 (sensitivity, specificity, F1).

Run from anywhere, on the machine that generated the published figures:
    python Review/additional_metrics.py

Two blocks are produced:

  (a) Figure 3B, lesion-proximal versus lesion-distal relative error. The drilled
      cavity is recovered as an interior hole in the bone mask, dilated to give a
      ring of neighbouring windows, and the error inside that ring is compared
      with the error elsewhere in the same slice.

  (b) Test-set metrics for both prediction strategies: MAE and RMSE in
      microstrain, per-slice spatial correlation, and classification of
      high-strain windows at the 10000 microstrain threshold.

The displacement-derived field is reported twice, once as the manuscript
currently computes it and once keeping it in its own physical units. The two
differ because DisplacementModel standardises with its own mean and standard
deviation while TrainingAnalysis de-standardises with the measured strain's.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom, binary_dilation, binary_fill_holes
from scipy.stats import pearsonr

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
YIELD_THRESHOLD = 10000          # microstrain
FIG3B_TEST_INDEX = 9             # plot_num = [3, 9]; index 3 is 3A, index 9 is 3B
LESION_RING_ITERATIONS = 2       # windows outward from the cavity edge


def aligned_stems():
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
    """High-strain classification at the yield threshold, on bone windows."""
    actual = np.abs(target) >= YIELD_THRESHOLD
    flagged = np.abs(predicted) >= YIELD_THRESHOLD
    tp = int(np.sum(actual & flagged))
    fp = int(np.sum(~actual & flagged))
    fn = int(np.sum(actual & ~flagged))
    tn = int(np.sum(~actual & ~flagged))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else float("nan")
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sens": sens, "spec": spec, "prec": prec, "f1": f1}


def field_metrics(target, predicted, mask_flat):
    """MAE and RMSE in microstrain, plus per-slice spatial correlation."""
    diffs, correlations = [], []
    for i in range(target.shape[0]):
        valid = mask_flat[i].astype(bool)
        if valid.sum() < 3:
            continue
        t, p = target[i][valid], predicted[i][valid]
        diffs.append(t - p)
        if np.std(t) > 0 and np.std(p) > 0:
            correlations.append(pearsonr(t, p)[0])
    d = np.concatenate(diffs)
    return {"MAE": float(np.mean(np.abs(d))),
            "RMSE": float(np.sqrt(np.mean(d ** 2))),
            "r_mean": float(np.mean(correlations)) if correlations else float("nan"),
            "r_median": float(np.median(correlations)) if correlations else float("nan"),
            "n_slices": len(correlations)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig3b", default=None,
                        help="filename (no .tif) of the Figure 3B case; "
                             "defaults to the original file order")
    args = parser.parse_args()

    stems = aligned_stems()
    test_stems = legacy_test_stems()
    fig3b = args.fig3b or test_stems[FIG3B_TEST_INDEX]
    print(f"{len(stems)} aligned slices, {len(test_stems)} in the test split")
    print(f"Figure 3B case: {fig3b}\n")

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
    print(f"target mean={mean_S:.2f} std={std_S:.2f} | "
          f"derived mean={mean_D:.2f} std={std_D:.2f} | "
          f"scale factor={std_S / std_D:.4f}\n")

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

    # ---- (a) Figure 3B: relative error close to the lesion -------------------
    print("=" * 82)
    print(f"(a) Figure 3B, relative error near the lesion   [{fig3b}]")
    print("=" * 82)

    def find_cavity(idx):
        """
        Recover the drilled defect as an enclosed hole in the bone mask.

        The masks are natively 20x20, so a 5 mm cavity spans only two or three
        windows. be_mask is dilated then eroded twice, which can close a hole
        that small, so detect on the raw mask and intersect afterwards.
        """
        raw = masks[idx].astype(bool)
        cav = binary_fill_holes(raw) & ~raw
        if cav.any():
            return cav, "raw mask"
        eroded = be_mask[idx].astype(bool)
        cav = binary_fill_holes(eroded) & ~eroded
        return (cav, "eroded mask") if cav.any() else (None, None)

    def show(grid, title):
        print(f"  {title}")
        for row in grid:
            print("    " + "".join("#" if v else "." for v in row))

    def lesion_report(idx, stem):
        cavity, source = find_cavity(idx)
        bone = be_mask[idx].astype(bool)
        t = target[idx]
        if cavity is None:
            print(f"\n{stem}: no enclosed cavity found in either mask.")
            show(masks[idx].astype(bool), "raw mask")
            show(bone, "eroded mask (be_mask)")
            return None
        ring = binary_dilation(cavity, iterations=LESION_RING_ITERATIONS) & bone & ~cavity
        distal = bone & ~ring & ~cavity
        print(f"\n{stem}  (cavity from {source}: {int(cavity.sum())} windows, "
              f"proximal {int(ring.sum())}, distal {int(distal.sum())})")
        if ring.sum() == 0 or distal.sum() == 0:
            print("  too few windows on one side to compare")
            return None
        print(f"  {'field':<24}{'proximal med':>14}{'distal med':>13}"
              f"{'difference':>13}{'ratio':>9}")
        out = {}
        for label, field in fields.items():
            err = relative_error(t, field[idx]).reshape(20, 20)
            sel_p = ring & (t.reshape(20, 20) != 0)
            sel_d = distal & (t.reshape(20, 20) != 0)
            if sel_p.sum() == 0 or sel_d.sum() == 0:
                continue
            mp, md = float(np.median(err[sel_p])), float(np.median(err[sel_d]))
            ratio = mp / md if md else float("nan")
            print(f"  {label:<24}{mp:>14.1f}{md:>13.1f}{mp - md:>13.1f}{ratio:>9.2f}")
            out[label] = (mp, md)
        return out

    lesion_report(index[fig3b], fig3b)

    print("\n" + "-" * 82)
    print("Same comparison for every lesioned slice in the test split")
    print("-" * 82)
    for stem in test_stems:
        if "_LES_" in stem and stem in index and stem != fig3b:
            lesion_report(index[stem], stem)

    # ---- (b) Additional test-set metrics -------------------------------------
    print("\n" + "=" * 82)
    print("(b) Test-set metrics over bone windows")
    print("=" * 82)
    print(f"\n{'field':<24}{'MAE (ue)':>12}{'RMSE (ue)':>12}"
          f"{'mean r':>10}{'median r':>11}{'slices':>8}")
    print("-" * 82)
    for label, field in fields.items():
        m = field_metrics(target[test_idx], field[test_idx], mask_flat[test_idx])
        print(f"{label:<24}{m['MAE']:>12.1f}{m['RMSE']:>12.1f}"
              f"{m['r_mean']:>10.3f}{m['r_median']:>11.3f}{m['n_slices']:>8}")
    print("-" * 82)

    print(f"\nHigh-strain classification at {YIELD_THRESHOLD} microstrain")
    print(f"{'field':<24}{'TP':>7}{'FP':>7}{'FN':>7}{'TN':>7}"
          f"{'sens':>9}{'spec':>9}{'prec':>9}{'F1':>9}")
    print("-" * 82)
    valid = mask_flat[test_idx].astype(bool)
    for label, field in fields.items():
        c = classification(target[test_idx][valid], field[test_idx][valid])
        print(f"{label:<24}{c['TP']:>7}{c['FP']:>7}{c['FN']:>7}{c['TN']:>7}"
              f"{c['sens']:>9.3f}{c['spec']:>9.3f}{c['prec']:>9.3f}{c['f1']:>9.3f}")
    print("-" * 82)

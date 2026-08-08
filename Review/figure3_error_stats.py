"""
Summary statistics of the relative strain error for the two cases shown in Figure 3.

Addresses Reviewer 1, Comment 3: provides a quantitative comparison of the
displacement-derived (D2IM) and directly predicted (D2IM-Strain) strain fields.

Run from anywhere:
    python Review/figure3_error_stats.py

Figure 3 corresponds to test indices 3 and 9, as set by `plot_num` in
TrainingAnalysis.visualise_strain().
"""

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

from LoadingData.ImageResizer import ImageResizer
from DataProcessing.Masking import Masking
from DataProcessing.Strain import Strain
from Training.DataSpilt import DataSplit

# Test indices plotted in Figure 3 (see TrainingAnalysis.visualise_strain)
FIG3_INDICES = [9, 3]  # 9 = Fig 3A (intact), 3 = Fig 3B (lesioned)

NODE_SPACING = 50  # Node spacing, as used in Strain and DisplacementModel

DISPLACEMENT_MODEL = PROJECT_ROOT / "Main" / "D2IM_trained.h5"
DIRECT_MODEL = PROJECT_ROOT / "Main" / "M1_best.h5"


def relative_error(target, predicted):
    """Relative strain error in %, matching TrainingAnalysis.visualise_strain."""
    error = np.abs((target - predicted) / target) * 100
    return np.nan_to_num(error, posinf=0, neginf=0)


def summarise(error, valid):
    """Summary statistics over the valid (bone, non-zero target) cells only."""
    vals = error[valid]
    return {
        "n": vals.size,
        "mean": np.mean(vals),
        "sd": np.std(vals),
        "median": np.median(vals),
        "p95": np.percentile(vals, 95),
        "max": np.max(vals),
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
    # ---- Load and resize, exactly as in Main.py -------------------------------
    resizer = ImageResizer()
    all_loaders = resizer.get_resized_loaders()

    scan_array = np.array([il.image for il in all_loaders[0].images]) / 255
    mask_array = [il.image for il in all_loaders[1].images]
    input_w_array = [il.image for il in all_loaders[2].images]
    scan_filenames = [il.metadata["filename"] for il in all_loaders[0].images]

    mask_obj = Masking(mask_array)
    be_mask = mask_obj.get_binary_erosion_mask()
    bd_mask = mask_obj.get_binary_dilation_mask()

    # ---- Ground-truth strain in microstrain ----------------------------------
    strain_obj = Strain(be_mask, input_w_array)
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
    # Both variants are reported below so the effect can be judged.
    global_mean_D, global_std_D = np.mean(derived_ezz), np.std(derived_ezz)
    standardized_pezz = np.where(be_mask, (derived_ezz - global_mean_D) / global_std_D, 0.0)

    print(f"\ntarget  strain: mean={global_mean_S:.2f}, std={global_std_S:.2f}")
    print(f"derived strain: mean={global_mean_D:.2f}, std={global_std_D:.2f}")
    print(f"scale factor applied by the as-published path: {global_std_S / global_std_D:.4f}")

    # ---- Split, exactly as in Main.py ----------------------------------------
    _, _, scan_test = DataSplit(scan_array).split_data()
    _, _, mask_test = DataSplit(be_mask).split_data()
    _, _, strain_test = DataSplit(strain_obj.standardized_ezz).split_data()
    _, _, derived_raw_test = DataSplit(derived_ezz).split_data()
    _, _, pezz_test = DataSplit(standardized_pezz).split_data()
    _, _, filenames_test = DataSplit(scan_filenames).split_data()

    # ---- Direct strain prediction (D2IM-Strain) ------------------------------
    direct_model = tf.keras.models.load_model(DIRECT_MODEL)
    n = scan_test.shape[0]
    input_scan = scan_test.reshape(n, scan_test.shape[1], scan_test.shape[2], 1)
    input_mask = mask_test.reshape(n, mask_test.shape[1], mask_test.shape[2], 1)
    direct_predictions = direct_model.predict([input_scan, input_mask])

    # ---- De-standardise back to microstrain ----------------------------------
    mask_flat = mask_test.reshape(n, 400)
    target = np.where(mask_flat, strain_test.reshape(n, 400) * global_std_S + global_mean_S, 0.0)
    direct = np.where(mask_flat, direct_predictions * global_std_S + global_mean_S, 0.0)

    # As published: derived field de-standardised with the TARGET's mean/std
    derived_as_published = np.where(
        mask_flat, pezz_test.reshape(n, 400) * global_std_S + global_mean_S, 0.0)

    # Corrected: derived field left in its own physical units
    derived_corrected = np.where(mask_flat, derived_raw_test.reshape(n, 400), 0.0)

    # ---- Statistics for the two Figure 3 cases -------------------------------
    for idx in FIG3_INDICES:
        panel = "Fig 3A" if idx == 9 else "Fig 3B"
        lesion = "LESION" if "LES" in filenames_test[idx].upper() else "intact"
        print(f"\n\n{'=' * 78}")
        print(f"{panel}  |  test index {idx}  |  {filenames_test[idx]}  ({lesion})")
        print("=" * 78)

        t = target[idx]
        # Restrict to bone cells with a non-zero target, so the background
        # (which is identically zero) does not dilute the statistics.
        valid = mask_flat[idx].astype(bool) & (t != 0)

        rows = [
            ("derived (as published)", summarise(relative_error(t, derived_as_published[idx]), valid)),
            ("derived (corrected)", summarise(relative_error(t, derived_corrected[idx]), valid)),
            ("direct (D2IM-Strain)", summarise(relative_error(t, direct[idx]), valid)),
        ]
        print_table("Relative strain error (%), bone cells only", rows)

        # Full-field figures, including background, for comparison with the
        # error maps in Figure 3, which are computed over the whole 20x20 grid.
        all_cells = np.ones_like(t, dtype=bool)
        rows_full = [
            ("derived (as published)", summarise(relative_error(t, derived_as_published[idx]), all_cells)),
            ("derived (corrected)", summarise(relative_error(t, derived_corrected[idx]), all_cells)),
            ("direct (D2IM-Strain)", summarise(relative_error(t, direct[idx]), all_cells)),
        ]
        print_table("Relative strain error (%), full 20x20 field (as plotted)", rows_full)

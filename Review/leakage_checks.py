"""
How much of the test accuracy the slice-level split could explain.

Addresses Reviewer 2 in the second round, who argues that having slices of one
vertebra in both the training and the test set lets the model recall
near-duplicates. Four checks, reported here as four blocks:

  (a) Slices per specimen and slicing direction, how they fall across the
      training, validation and test sets, and the test accuracy per specimen.

  (b) For every test slice, how many of its two same-direction neighbours are in
      the training set, with the accuracy for 0, 1 and 2. Two further columns
      give the fuller picture: the crossing slices of the same specimen that are
      in training (each shares one line of DVC nodes with the test slice), and
      whether the same plane of the paired scan is in training.

  (c) How alike neighbouring slices are, against the same plane in another
      vertebra, for the input image and for the measured strain field. Also
      saves an example figure for one intact and one lesioned test slice.

  (d) The lesioned test slices: how many of their neighbours are in training,
      how many training slices show a lesion at the same site in another
      vertebra, and how well each is predicted. The cavity is drilled through
      the vertebra, so it appears in neighbouring planes. It is not located
      from the mask, because that detection is not specific to lesioned slices.

Run from anywhere, on the machine that generated the published figures:
    python Review/leakage_checks.py

Only the direct model is evaluated. The split is taken over the filenames in
alphabetical order, which gives the same number of test slices per specimen as
the published run on any machine. The model file is chosen by checking each
candidate against the R2 values of Table 2, so the script either reproduces the
published numbers first or stops.
"""

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom, gaussian_filter
from sklearn.metrics import r2_score

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
OUT_DIR = PROJECT_ROOT / "Review" / "output"

# The selected model of Table 2 (MAE, dropout 0.2, L2 1e-6). It is Main/M1_best.h5
# on the machine that produced the figures and Models/mae-02-e6-055.h5 elsewhere.
# A file named M1_best.h5 is not that model on every machine, so each candidate
# is checked against Table 2 before it is used.
MODEL_CANDIDATES = [PROJECT_ROOT / "Main" / "M1_best.h5",
                    PROJECT_ROOT / "Models" / "mae-02-e6-055.h5"]
TABLE_2_R2 = (0.74, 0.70, 0.55)  # train, validation, test

TEXTURE_SIGMA = 8                # pixels of the 256 x 256 input
RANDOM_PAIRS = 1000

# Lesion site per lesioned specimen, from the label mapping in
# TrainingAnalysis.visualise_correlation.
LESION_SITE = {"S2": "anterior", "S8": "anterior", "S10": "anterior",
               "S4": "lateral", "S6": "lateral"}

# Test slices per specimen in the published run, from the same mapping.
PUBLISHED_TEST_COUNTS = {"S1": 6, "S2": 3, "S3": 0, "S4": 2, "S5": 1,
                         "S6": 2, "S7": 2, "S8": 5, "S9": 3, "S10": 2}

NAME = re.compile(r"^(S\d+)_(INT|LES)_UL_(AP|ML)_\d+_(\d+)$")


def parse(stem):
    """Specimen, INT or LES, slicing direction and node-plane index."""
    specimen, kind, direction, plane = NAME.match(stem).groups()
    return specimen, kind, direction, int(plane)


def vertebra_of(specimen):
    """Labels alternate INT/LES: S1 and S2 are one bone before and after drilling."""
    return (int(specimen[1:]) + 1) // 2


def partner_of(specimen):
    number = int(specimen[1:])
    return f"S{number + 1 if number % 2 else number - 1}"


def aligned_stems():
    """Filenames common to all three folders. The folders differ in size."""
    sets = [{p.stem for p in d.glob("*.tif")} for d in (SCAN_DIR, MASK_DIR, W_DIR)]
    return sorted(sets[0] & sets[1] & sets[2])


def published_split():
    """
    Training, validation and test filenames of the published run.

    FolderImageLoader takes the files in whatever order the operating system
    lists them, which differs between machines. Alphabetical order reproduces
    the published number of test slices for every specimen.
    """
    order = sorted(p.stem for p in SCAN_DIR.glob("*.tif"))
    train, val, test = DataSplit(order).split_data()
    return list(train), list(val), list(test)


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


def accuracy(t, p):
    """R2, median relative error (%) and MAE (microstrain) over a set of windows."""
    if t.size < 2 or np.var(t) == 0:
        return {"n": int(t.size), "r2": float("nan"), "rel": float("nan"), "mae": float("nan")}
    return {"n": int(t.size), "r2": float(r2_score(t, p)),
            "rel": float(np.median(relative_error(t, p))),
            "mae": float(np.mean(np.abs(t - p)))}


def correlation(a, b):
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def quartiles(values):
    v = np.array([x for x in values if np.isfinite(x)])
    if v.size == 0:
        return f"{'n/a':>22}"
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return f"{med:6.2f} [{q1:5.2f}, {q3:5.2f}]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intact", default=None,
                        help="filename (no .tif) of the intact test slice for the figure")
    parser.add_argument("--lesioned", default=None,
                        help="filename (no .tif) of the lesioned test slice for the figure")
    args = parser.parse_args()

    stems = aligned_stems()
    train_stems, val_stems, test_stems = published_split()
    where = {}
    for name, group in (("train", train_stems), ("val", val_stems), ("test", test_stems)):
        for s in group:
            where[s] = name
    print(f"{len(stems)} aligned slices; split {len(train_stems)} / {len(val_stems)} / "
          f"{len(test_stems)} (train / val / test)")
    test_counts = {sp: sum(s.startswith(sp + "_") for s in test_stems)
                   for sp in PUBLISHED_TEST_COUNTS}
    print("test slices per specimen match the published run: "
          f"{'yes' if test_counts == PUBLISHED_TEST_COUNTS else 'NO, do not use these numbers'}")

    scan = load_stack(SCAN_DIR, stems, 256) / 255
    masks = load_stack(MASK_DIR, stems, 20)
    w = load_stack(W_DIR, stems, 20)

    mask_obj = Masking(masks)
    be_mask = mask_obj.get_binary_erosion_mask()

    strain_obj = Strain(be_mask, w)
    mean_S, std_S = strain_obj.global_mean, strain_obj.global_std

    n = len(stems)
    mask_flat = be_mask.reshape(n, 400)
    target = np.where(mask_flat, strain_obj.standardized_ezz.reshape(n, 400) * std_S + mean_S, 0.0)
    valid = mask_flat.astype(bool) & (target != 0)

    index = {s: i for i, s in enumerate(stems)}

    def set_r2(field):
        """R2 over the training, validation and test slices."""
        scores = []
        for group in (train_stems, val_stems, test_stems):
            idx = [index[s] for s in group if s in index]
            scores.append(accuracy(target[idx][valid[idx]], field[idx][valid[idx]])["r2"])
        return scores

    direct = None
    for candidate in MODEL_CANDIDATES:
        if not candidate.exists():
            continue
        prediction = tf.keras.models.load_model(candidate).predict(
            [scan.reshape(-1, scan.shape[1], scan.shape[2], 1),
             be_mask.reshape(-1, 20, 20, 1)], verbose=0)
        field = np.where(mask_flat, prediction * std_S + mean_S, 0.0)
        scores = set_r2(field)
        matches = all(abs(a - b) < 0.015 for a, b in zip(scores, TABLE_2_R2))
        print(f"model {candidate.parent.name}/{candidate.name}: R2 train / val / test = "
              + " / ".join(f"{r:.2f}" for r in scores)
              + ("  (matches Table 2)" if matches else "  (does not match Table 2, skipped)"))
        if matches:
            direct = field
            break
    if direct is None:
        sys.exit("No model file reproduces Table 2 (0.74 / 0.70 / 0.55). Put the selected model at "
                 "Main/M1_best.h5 or Models/mae-02-e6-055.h5 and run again.")
    info = {s: parse(s) for s in stems}
    at = {(sp, d, pl): s for s, (sp, _, d, pl) in info.items()}
    test_in = sorted(s for s in test_stems if s in index)
    specimens = sorted({info[s][0] for s in stems}, key=lambda k: int(k[1:]))

    def status(specimen, direction, plane):
        """Which set the slice at this node plane is in, or '-' if it was not retained."""
        stem = at.get((specimen, direction, plane))
        return where.get(stem, "-") if stem else "-"

    def pooled(stem_list):
        idx = [index[s] for s in stem_list]
        v = valid[idx]
        return accuracy(target[idx][v], direct[idx][v])

    # ---- (a) slices and test accuracy per specimen ---------------------------
    print("\n" + "=" * 107)
    print("(a) Slices per specimen and direction, the split, and test accuracy (direct model)")
    print("=" * 107)
    print(f"{'specimen':<10}{'type':<10}{'AP':>4}{'ML':>4}{'all':>5}"
          f"{'train':>7}{'val':>5}{'test':>6}{'test AP':>9}{'test ML':>9}"
          f"{'windows':>9}{'R2':>8}{'med rel %':>11}{'MAE (ue)':>10}")
    print("-" * 107)
    for sp in specimens:
        members = [s for s in stems if info[s][0] == sp]
        kind = LESION_SITE.get(sp, "intact")
        by_set = {k: [s for s in members if where.get(s) == k] for k in ("train", "val", "test")}
        acc = pooled(by_set["test"]) if by_set["test"] else {"n": 0, "r2": float("nan"),
                                                               "rel": float("nan"), "mae": float("nan")}
        print(f"{sp:<10}{kind:<10}"
              f"{sum(info[s][2] == 'AP' for s in members):>4}"
              f"{sum(info[s][2] == 'ML' for s in members):>4}{len(members):>5}"
              f"{len(by_set['train']):>7}{len(by_set['val']):>5}{len(by_set['test']):>6}"
              f"{sum(info[s][2] == 'AP' for s in by_set['test']):>9}"
              f"{sum(info[s][2] == 'ML' for s in by_set['test']):>9}"
              f"{acc['n']:>9}{acc['r2']:>8.2f}{acc['rel']:>11.1f}{acc['mae']:>10.0f}")
    print("-" * 107)
    acc = pooled(test_in)
    print(f"{'all test':<20}{'':>13}{'':>12}{len(test_in):>6}{'':>18}"
          f"{acc['n']:>9}{acc['r2']:>8.2f}{acc['rel']:>11.1f}{acc['mae']:>10.0f}")

    # ---- (b) neighbours of each test slice -----------------------------------
    print("\n" + "=" * 104)
    print("(b) Test slices by the number of same-direction neighbours in the training set")
    print("=" * 104)
    print("prev / next: the slice one node plane either side, same scan and direction")
    print("nearest: planes to the closest training slice of the same scan and direction")
    print("crossing: slices of the same scan cut in the other direction that are in training")
    print("paired: the same plane of the paired scan (same bone, intact against lesioned)\n")
    print(f"{'test slice':<26}{'prev':>7}{'next':>7}{'in train':>10}{'nearest':>9}"
          f"{'crossing':>10}{'paired':>8}{'windows':>9}{'R2':>8}{'med rel %':>11}")
    print("-" * 104)
    rows = []
    for s in test_in:
        sp, _, d, pl = info[s]
        other = "ML" if d == "AP" else "AP"
        prev, nxt = status(sp, d, pl - 1), status(sp, d, pl + 1)
        in_train = [prev, nxt].count("train")
        train_planes = [p for (a, b, p), t in at.items()
                        if a == sp and b == d and where.get(t) == "train"]
        nearest = min((abs(p - pl) for p in train_planes), default=None)
        crossing = [t for (a, b, _), t in at.items() if a == sp and b == other]
        crossing_train = sum(where.get(t) == "train" for t in crossing)
        acc = pooled([s])
        rows.append((in_train, s, acc))
        print(f"{s:<26}{prev:>7}{nxt:>7}{in_train:>10}"
              f"{nearest if nearest is not None else '-':>9}"
              f"{f'{crossing_train}/{len(crossing)}':>10}{status(partner_of(sp), d, pl):>8}"
              f"{acc['n']:>9}{acc['r2']:>8.2f}{acc['rel']:>11.1f}")
    print("-" * 104)
    print(f"\n{'neighbours in train':<22}{'slices':>8}{'windows':>9}{'pooled R2':>11}"
          f"{'med rel %':>11}{'MAE (ue)':>10}{'median slice R2':>17}")
    print("-" * 88)
    for k in (0, 1, 2):
        group = [s for count, s, _ in rows if count == k]
        if not group:
            print(f"{k:<22}{0:>8}")
            continue
        acc = pooled(group)
        slice_r2 = [a["r2"] for count, _, a in rows if count == k and np.isfinite(a["r2"])]
        print(f"{k:<22}{len(group):>8}{acc['n']:>9}{acc['r2']:>11.2f}{acc['rel']:>11.1f}"
              f"{acc['mae']:>10.0f}{np.median(slice_r2) if slice_r2 else float('nan'):>17.2f}")
    print("-" * 88)

    # ---- (c) how alike neighbouring slices are --------------------------------
    print("\n" + "=" * 104)
    print(f"(c) Similarity between pairs of slices cut in the same direction (all {n} slices)")
    print("=" * 104)
    bone256 = load_stack(MASK_DIR, stems, 256).astype(bool)
    texture = np.array([img - gaussian_filter(img, TEXTURE_SIGMA) for img in scan])
    valid2d = valid.reshape(n, 20, 20)
    target2d = target.reshape(n, 20, 20)

    def similarity(i, j):
        """Correlation of the input image, its trabecular texture and the strain field."""
        out = [float("nan")] * 3
        region = bone256[i] & bone256[j]
        if region.sum() >= 100:
            out[0] = correlation(scan[i][region], scan[j][region])
            out[1] = correlation(texture[i][region], texture[j][region])
        shared = valid2d[i] & valid2d[j]
        if shared.sum() >= 20:
            out[2] = correlation(target2d[i][shared], target2d[j][shared])
        return out

    pairs = {"next plane, same scan": [], "two planes apart, same scan": [],
             "same bone, intact against lesioned": [], "another vertebra, same plane": [],
             "another vertebra, any plane": []}
    for i in range(n):
        sp_i, _, d_i, pl_i = info[stems[i]]
        for j in range(i + 1, n):
            sp_j, _, d_j, pl_j = info[stems[j]]
            if d_i != d_j:
                continue
            if sp_i == sp_j:
                if abs(pl_i - pl_j) == 1:
                    pairs["next plane, same scan"].append((i, j))
                elif abs(pl_i - pl_j) == 2:
                    pairs["two planes apart, same scan"].append((i, j))
            elif vertebra_of(sp_i) == vertebra_of(sp_j):
                if pl_i == pl_j:
                    pairs["same bone, intact against lesioned"].append((i, j))
            elif pl_i == pl_j:
                pairs["another vertebra, same plane"].append((i, j))
            else:
                pairs["another vertebra, any plane"].append((i, j))
    rng = np.random.default_rng(0)
    anywhere = pairs["another vertebra, any plane"]
    if len(anywhere) > RANDOM_PAIRS:
        chosen = rng.choice(len(anywhere), RANDOM_PAIRS, replace=False)
        pairs["another vertebra, any plane"] = [anywhere[k] for k in chosen]

    print("Median [lower quartile, upper quartile] of the correlation coefficient.")
    print(f"Texture: the input after subtracting a Gaussian background, sigma {TEXTURE_SIGMA} px.\n")
    print(f"{'pair':<38}{'pairs':>7}{'input image':>24}{'texture':>24}{'measured strain':>24}")
    print("-" * 117)
    for name, members in pairs.items():
        scores = np.array([similarity(i, j) for i, j in members]) if members else np.empty((0, 3))
        print(f"{name:<38}{len(members):>7}"
              + "".join(f"{quartiles(scores[:, c]):>24}" for c in range(3)))
    print("-" * 117)

    def example_figure(stem):
        sp, kind, d, pl = info[stem]
        others = [at[(o, d, pl)] for o in specimens
                  if vertebra_of(o) != vertebra_of(sp) and (o, d, pl) in at
                  and info[at[(o, d, pl)]][1] == kind]
        panels = [(at.get((sp, d, pl - 1)), "previous plane"), (stem, "test slice"),
                  (at.get((sp, d, pl + 1)), "next plane"),
                  (others[0] if others else None, "another vertebra, same plane")]
        own = [index[s] for s, _ in panels[:3] if s]
        limit = np.percentile(np.abs(target2d[own][valid2d[own]]), 98)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for col, (s, title) in enumerate(panels):
            for row in (0, 1):
                axes[row, col].axis("off")
            if s is None:
                axes[0, col].set_title(f"{title}\nnot in the dataset", fontsize=10)
                continue
            i = index[s]
            r = similarity(index[stem], i)
            note = "" if s == stem else f"\ntexture r = {r[1]:.2f}, strain r = {r[2]:.2f}"
            axes[0, col].imshow(scan[i], cmap="gray")
            axes[0, col].set_title(f"{title} ({where.get(s, '-')})\n{s}{note}", fontsize=10)
            im = axes[1, col].imshow(np.where(valid2d[i], target2d[i], np.nan),
                                     cmap="coolwarm", vmin=-limit, vmax=limit)
        fig.colorbar(im, ax=axes[1, :], shrink=0.8, label="measured ezz (microstrain)")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"neighbours_{stem}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")

    def has_both_neighbours(s):
        sp, _, d, pl = info[s]
        return (sp, d, pl - 1) in at and (sp, d, pl + 1) in at

    print()
    for kind, chosen in (("INT", args.intact), ("LES", args.lesioned)):
        candidates = [s for s in test_in if info[s][1] == kind and has_both_neighbours(s)]
        stem = chosen or (candidates or [None])[0]
        if stem is None or stem not in index:
            print(f"no {kind} test slice available for the figure")
            continue
        example_figure(stem)

    # ---- (d) lesioned test slices --------------------------------------------
    print("\n" + "=" * 104)
    print("(d) Lesioned test slices, their neighbours and how they are predicted")
    print("=" * 104)
    print("in train: same-direction neighbours of the same scan that are in training")
    print("same site: training slices, same direction, with a lesion at that site in another vertebra\n")
    print(f"{'test slice':<26}{'site':<10}{'in train':>10}{'same site':>11}"
          f"{'windows':>9}{'R2':>8}{'med rel %':>11}")
    print("-" * 85)
    by_stem = {s: (count, acc) for count, s, acc in rows}
    for s in test_in:
        sp, kind, d, _ = info[s]
        if kind != "LES":
            continue
        same_site = sum(1 for t in stems
                        if where.get(t) == "train" and info[t][1] == "LES" and info[t][2] == d
                        and info[t][0] != sp and LESION_SITE[info[t][0]] == LESION_SITE[sp])
        count, acc = by_stem[s]
        print(f"{s:<26}{LESION_SITE[sp]:<10}{count:>10}{same_site:>11}"
              f"{acc['n']:>9}{acc['r2']:>8.2f}{acc['rel']:>11.1f}")
    print("-" * 85)
    print(f"\n{'group':<34}{'slices':>8}{'pooled R2':>11}{'median slice R2':>17}")
    print("-" * 70)
    groups = [(f"lesioned, {k} in train", [s for c, s, _ in rows if info[s][1] == "LES" and c == k])
              for k in (0, 1, 2)]
    groups += [("lesioned, all", [s for _, s, _ in rows if info[s][1] == "LES"]),
               ("intact, all", [s for _, s, _ in rows if info[s][1] == "INT"])]
    for name, group in groups:
        if not group:
            print(f"{name:<34}{0:>8}")
            continue
        slice_r2 = [by_stem[s][1]["r2"] for s in group if np.isfinite(by_stem[s][1]["r2"])]
        print(f"{name:<34}{len(group):>8}{pooled(group)['r2']:>11.2f}"
              f"{np.median(slice_r2) if slice_r2 else float('nan'):>17.2f}")
    print("-" * 70)

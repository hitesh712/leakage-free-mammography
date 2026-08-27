"""Sensitivity of the patch classifier to imperfect lesion localisation.

Reviewer 5, comment 1: the pipeline assumes the lesion centre / bounding box is
known exactly from metadata, whereas a deployed system would receive noisy boxes
from a detector (CADe). This perturbs each crop by a random translation, given
as a fraction of the patch side, and by a scale factor, then re-evaluates the
already-trained model. No retraining -- this measures deployment-time
robustness, not adaptability.

Because the perturbed crops are regenerated from the source images rather than
taken from the official pre-cropped ROIs, the shift = 0 row is the correct
baseline for judging degradation; it will not exactly equal the headline AUC.

Usage:  python revision/rev06_perturbation.py [--seeds 3] [--repeats 3]
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

import revlib
from prep_data import inbreast_crop, mias_crop, prep, read_dcm

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "out")
WEIGHTS = os.path.join(HERE, "weights")
MIAS_DIR = "D:/all-mias/"
PATCH = 224

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--repeats", type=int, default=3, help="random jitter draws per sample per level")
ap.add_argument("--shifts", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20])
ap.add_argument("--scales", type=float, nargs="+", default=[0.85, 1.15])
args = ap.parse_args()
SEEDS = revlib.SEEDS[: args.seeds]

mi = pd.read_csv(os.path.join(CACHE, "mias_manifest.csv"))
inb = pd.read_csv(os.path.join(CACHE, "inbreast_manifest.csv"))
cb = pd.read_csv(os.path.join(CACHE, "cbis_manifest.csv"))
cbte = cb[(cb["split"] == "test") & cb["mask"].notna() & cb["full"].notna()].reset_index(drop=True)
log(f"MIAS {len(mi)} | INbreast {len(inb)} | CBIS test with mask+full {len(cbte)}")

import tensorflow as tf
from sklearn.metrics import roc_auc_score

log(f"TF {tf.__version__} | GPUs {[g.name for g in tf.config.list_physical_devices('GPU')]}")

models = []
for s in SEEDS:
    m, _ = revlib.load_effb0(tf, os.path.join(WEIGHTS, f"effb0_seed{s}.weights.h5"))
    models.append(m)
log(f"loaded {len(models)} trained models")


def ensemble_prob(X):
    """Mean sigmoid over the seed ensemble."""
    B = revlib.as_batch(X)
    return np.mean([m.predict(B, batch_size=32, verbose=0).ravel() for m in models], axis=0)


# --------------------------------------------------------------------------
# source images, loaded once
# --------------------------------------------------------------------------
MIAS_IMG = {r: cv2.imread(MIAS_DIR + r + ".pgm", cv2.IMREAD_GRAYSCALE) for r in mi["refnum"].unique()}
INB_IMG = {}
for d in inb["dcm"].unique():
    INB_IMG[d] = read_dcm(d)
log("source images loaded (MIAS, INbreast)")


def cbis_bbox(mask_path):
    """Tight bounding box of the annotated lesion in a CBIS ROI mask."""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    ys, xs = np.where(m > 127)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), m.shape


def cbis_crop(full_img, bbox, mask_shape, size, shift=(0.0, 0.0), scale=1.0):
    """Crop the lesion from the full mammogram, rescaling mask coords if needed."""
    H, W = full_img.shape
    mh, mw = mask_shape
    sx, sy = W / mw, H / mh
    x0, y0, x1, y1 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(np.clip(max(x1 - x0, y1 - y0) * 1.3 * scale, 32, min(H, W)))
    half = side // 2
    cx = int(np.clip(round(cx + shift[0] * side), half, W - half))
    cy = int(np.clip(round(cy + shift[1] * side), half, H - half))
    return prep(full_img[cy - half:cy - half + side, cx - half:cx - half + side], size)


# Precompute CBIS geometry once (reading 399 full mammograms is the slow part).
cbis_geo = []
for _, r in cbte.iterrows():
    bb = cbis_bbox(r["mask"])
    if bb is None:
        continue
    cbis_geo.append((r["full"], bb[:4], bb[4], int(r["label"])))
log(f"CBIS test lesions with usable masks: {len(cbis_geo)}")


def eval_level(shift_frac, scale, rng):
    """Return AUC per dataset at one perturbation level, averaged over repeats."""
    out = {}

    # MIAS
    aucs = []
    for _ in range(args.repeats if shift_frac > 0 or scale != 1.0 else 1):
        X = []
        for _, r in mi.iterrows():
            th = rng.uniform(0, 2 * np.pi)
            sh = (shift_frac * np.cos(th), shift_frac * np.sin(th))
            p, _ = mias_crop(MIAS_IMG[r["refnum"]], r["x"], r["y"], r["r"], PATCH, sh, scale)
            X.append(p)
        aucs.append(roc_auc_score(mi["label"].values, ensemble_prob(np.array(X, np.uint8))))
    out["mias"] = float(np.mean(aucs))

    # INbreast
    aucs = []
    for _ in range(args.repeats if shift_frac > 0 or scale != 1.0 else 1):
        X = []
        for _, r in inb.iterrows():
            th = rng.uniform(0, 2 * np.pi)
            sh = (shift_frac * np.cos(th), shift_frac * np.sin(th))
            X.append(inbreast_crop(INB_IMG[r["dcm"]], (r["x0"], r["y0"], r["x1"], r["y1"]),
                                   PATCH, sh, scale))
        aucs.append(roc_auc_score(inb["label"].values, ensemble_prob(np.array(X, np.uint8))))
    out["inbreast"] = float(np.mean(aucs))

    # CBIS test
    aucs = []
    for _ in range(args.repeats if shift_frac > 0 or scale != 1.0 else 1):
        X, y = [], []
        cache = {}
        for full, bb, mshape, lab in cbis_geo:
            if full not in cache:
                cache[full] = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
            img = cache[full]
            if img is None:
                continue
            th = rng.uniform(0, 2 * np.pi)
            sh = (shift_frac * np.cos(th), shift_frac * np.sin(th))
            X.append(cbis_crop(img, bb, mshape, PATCH, sh, scale))
            y.append(lab)
        aucs.append(roc_auc_score(y, ensemble_prob(np.array(X, np.uint8))))
    out["cbis_test"] = float(np.mean(aucs))
    return out


rows = []
rng = np.random.default_rng(0)
for sh in args.shifts:
    r = eval_level(sh, 1.0, rng)
    r.update({"shift_pct": int(sh * 100), "scale": 1.0})
    rows.append(r)
    log(f"  shift {int(sh * 100):3d}%  scale 1.00 -> "
        f"CBIS {r['cbis_test']:.4f} | MIAS {r['mias']:.4f} | INb {r['inbreast']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "perturbation.csv"), index=False)

for sc in args.scales:
    r = eval_level(0.0, sc, rng)
    r.update({"shift_pct": 0, "scale": sc})
    rows.append(r)
    log(f"  shift   0%  scale {sc:.2f} -> "
        f"CBIS {r['cbis_test']:.4f} | MIAS {r['mias']:.4f} | INb {r['inbreast']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "perturbation.csv"), index=False)

P = pd.DataFrame(rows)
P.to_csv(os.path.join(OUT, "perturbation.csv"), index=False)
print(P.to_string(index=False))
log("DONE")

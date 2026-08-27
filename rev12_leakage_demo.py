"""The leakage demonstration itself: Mini-MIAS whole-image, leaky vs corrected.

This is the paper's headline claim (Table 2) and the one experiment for which no
runnable script survived: every MIAS script in the project implements the
leakage-free variant only. Without this, the contribution "we release scripts
that reproduce both the leaky and the corrected pipelines" is not met.

Task: whole-image normal vs abnormal on Mini-MIAS, deep features + RBF-SVM,
exactly the setup the inflated literature numbers come from. The two arms differ
in one thing only -- whether augmentation happens before or after the split.

  leaky      augment all 322 images (12 rotations), THEN split at random.
             Rotated copies of one film land on both sides.
  corrected  split by image ID FIRST, then augment the training side only.

Usage:  python revision/rev12_leakage_demo.py [--folds 5] [--rotations 12]
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MIAS_DIR = "D:/all-mias/"
SIZE = 224

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--rotations", type=int, default=36,
                help="36 matches the original leaky notebook (every 10 degrees)")
ap.add_argument("--task", choices=["bm", "nva"], default="bm",
                help="bm = benign vs malignant on the 119 abnormal films "
                     "(the task the original leaky pipeline used); "
                     "nva = normal vs abnormal on all 322 films")
args = ap.parse_args()
ANGLES = [i * (360 // args.rotations) for i in range(args.rotations)]

# --------------------------------------------------------------------------
# Mini-MIAS whole images, one row per film.
#   bm : keep only films with severity B or M   -> 119 films, 68 B / 51 M
#   nva: all films, NORM vs everything else     -> 322 films, 207 / 115
# --------------------------------------------------------------------------
rows, seen = [], set()
for line in open(MIAS_DIR + "Info1.txt"):
    q = line.strip().split()
    if len(q) < 3 or not q[0].startswith("mdb") or q[0] in seen:
        continue
    seen.add(q[0])
    sev = q[3] if len(q) > 3 else ""
    if args.task == "bm":
        if sev not in ("B", "M"):
            continue
        rows.append({"image_id": q[0], "cls": q[2], "label": 0 if sev == "B" else 1})
    else:
        rows.append({"image_id": q[0], "cls": q[2],
                     "label": 0 if q[2].upper() == "NORM" else 1})
D = pd.DataFrame(rows)
log(f"task = {args.task}, rotations = {len(ANGLES)}")
log(f"Mini-MIAS films: {len(D)} | normal {int((D.label == 0).sum())} "
    f"abnormal {int((D.label == 1).sum())} | majority-class accuracy "
    f"{max((D.label == 0).mean(), (D.label == 1).mean()):.3f}")

_clahe = cv2.createCLAHE(2.0, (8, 8))


def load(image_id):
    g = cv2.imread(MIAS_DIR + image_id + ".pgm", cv2.IMREAD_GRAYSCALE)
    return None if g is None else _clahe.apply(cv2.resize(g, (SIZE, SIZE)))


def rotate(img, ang):
    if ang == 0:
        return img
    M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)


BASE = {}
for iid in D["image_id"]:
    im = load(iid)
    if im is not None:
        BASE[iid] = im
D = D[D["image_id"].isin(BASE)].reset_index(drop=True)
log(f"loaded {len(D)} films")

# --------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

log(f"TF {tf.__version__}")
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(SIZE, SIZE, 3))
base.trainable = False
extractor = Sequential([base, GlobalAveragePooling2D()])


def features(image_ids, angles):
    """Deep features for every (image, angle) pair, with its source image id."""
    X, y, grp = [], [], []
    batch, meta = [], []

    def flush():
        if not batch:
            return
        arr = preprocess_input(np.stack(batch).astype("float32"))
        f = extractor.predict(arr, batch_size=32, verbose=0)
        X.append(f)
        for iid, lab in meta:
            y.append(lab)
            grp.append(iid)
        batch.clear()
        meta.clear()

    lab_of = dict(zip(D["image_id"], D["label"]))
    for iid in image_ids:
        for a in angles:
            batch.append(np.stack([rotate(BASE[iid], a).astype("float32")] * 3, -1))
            meta.append((iid, lab_of[iid]))
            if len(batch) >= 256:
                flush()
    flush()
    return np.vstack(X), np.array(y), np.array(grp)


def fit_eval(Xtr, ytr, Xte, yte):
    sc = StandardScaler()
    clf = SVC(C=50, gamma=1e-3, kernel="rbf", probability=True,
              class_weight="balanced", random_state=42)
    clf.fit(sc.fit_transform(Xtr), ytr)
    Xs = sc.transform(Xte)
    prob = clf.predict_proba(Xs)[:, 1]
    pred = clf.predict(Xs)
    return accuracy_score(yte, pred), roc_auc_score(yte, prob)


# --------------------------------------------------------------------------
# ARM 1 -- leaky: augment everything, then split at random over the augmented pool
# --------------------------------------------------------------------------
log(f"extracting features for all {len(D)} films x {len(ANGLES)} rotations ...")
Xa, ya, ga = features(D["image_id"].values, ANGLES)
log(f"augmented pool: {Xa.shape}")

leaky_acc, leaky_auc = [], []
for tr, te in StratifiedKFold(args.folds, shuffle=True, random_state=0).split(Xa, ya):
    a, u = fit_eval(Xa[tr], ya[tr], Xa[te], ya[te])
    leaky_acc.append(a)
    leaky_auc.append(u)
log(f"LEAKY      acc {np.mean(leaky_acc):.4f} +/- {np.std(leaky_acc):.4f} | "
    f"AUC {np.mean(leaky_auc):.4f}")

# --------------------------------------------------------------------------
# ARM 2 -- corrected: split by image id first; augment the training side only
# --------------------------------------------------------------------------
corr_acc, corr_auc = [], []
for tr_i, te_i in GroupKFold(args.folds).split(Xa, ya, groups=ga):
    tr_ids = np.unique(ga[tr_i])
    te_ids = np.unique(ga[te_i])
    assert not (set(tr_ids) & set(te_ids)), "image leaked across the split"
    tr_mask = np.isin(ga, tr_ids)                      # all rotations, train side
    te_mask = np.isin(ga, te_ids) & (np.arange(len(ga)) % len(ANGLES) == 0)
    a, u = fit_eval(Xa[tr_mask], ya[tr_mask], Xa[te_mask], ya[te_mask])
    corr_acc.append(a)
    corr_auc.append(u)
log(f"CORRECTED  acc {np.mean(corr_acc):.4f} +/- {np.std(corr_acc):.4f} | "
    f"AUC {np.mean(corr_auc):.4f} +/- {np.std(corr_auc):.4f}")

maj = float(max((D.label == 0).mean(), (D.label == 1).mean()))
res = {
    "task": args.task,
    "n_films": int(len(D)), "rotations": len(ANGLES), "folds": args.folds,
    "majority_class_accuracy": round(maj, 4),
    "leaky_accuracy": round(float(np.mean(leaky_acc)), 4),
    "leaky_accuracy_std": round(float(np.std(leaky_acc)), 4),
    "leaky_auc": round(float(np.mean(leaky_auc)), 4),
    "corrected_accuracy": round(float(np.mean(corr_acc)), 4),
    "corrected_accuracy_std": round(float(np.std(corr_acc)), 4),
    "corrected_auc": round(float(np.mean(corr_auc)), 4),
    "corrected_auc_std": round(float(np.std(corr_auc)), 4),
}
with open(os.path.join(OUT, f"leakage_demo_{args.task}.json"), "w") as f:
    json.dump(res, f, indent=2)
print()
print(json.dumps(res, indent=2))
log("DONE")

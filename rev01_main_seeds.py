"""Headline model, re-run over 10 seeds with per-sample predictions retained.

This reproduces the submitted configuration exactly -- EfficientNetB0 @224,
trained on the CBIS-DDSM official TRAIN split with a patient-wise 15% validation
carve-out, evaluated on the official TEST split and on MIAS / INbreast without
retraining -- but now writes out every individual prediction.

Retaining per-sample scores is what makes the following possible, none of which
could be done from the summary CSVs of the original submission:
  * confusion matrices generated from predictions rather than reconstructed
  * DeLong tests between datasets and configurations
  * calibration curves, Brier score and ECE
  * Youden's J threshold recalibration and cross-domain threshold transfer
  * bootstrap CIs at a stated resampling level (patch or patient)

Reviewer coverage: R1.1, R1.2, R1.6, R1.9, R2.2, R2.5, R2.7, R5.4.

Usage:  python revision/rev01_main_seeds.py [--seeds N] [--quick]
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "out")
WEIGHTS = os.path.join(HERE, "weights")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WEIGHTS, exist_ok=True)

PATCH, BATCH = 224, 16
# The first three are the seeds used in the original submission; the remainder
# extend the stability evidence from three runs to ten (reviewers 1.2 and 2.2).
ALL_SEEDS = [2021, 7, 123, 42, 2024, 5, 777, 31337, 99, 1234]

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--quick", action="store_true", help="2 epochs per phase, smoke test only")
args = ap.parse_args()
SEEDS = ALL_SEEDS[: args.seeds]

# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
cb = pd.read_csv(os.path.join(CACHE, "cbis_manifest.csv"))
Xcb = np.load(os.path.join(CACHE, "cbis_roi_224.npy"))
mi = pd.read_csv(os.path.join(CACHE, "mias_manifest.csv"))
Xmi = np.load(os.path.join(CACHE, "mias_roi_224.npy"))
inb = pd.read_csv(os.path.join(CACHE, "inbreast_manifest.csv"))
Xinb = np.load(os.path.join(CACHE, "inbreast_roi_224.npy"))

tr_mask = (cb["split"] == "train").values
te_mask = (cb["split"] == "test").values
log(f"CBIS {Xcb.shape} train {tr_mask.sum()} test {te_mask.sum()} | "
    f"MIAS {Xmi.shape} | INbreast {Xinb.shape}")

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

log(f"TF {tf.__version__} | GPUs {[g.name for g in tf.config.list_physical_devices('GPU')]}")

import cv2


def rot(g, a):
    if a == 0:
        return g
    return cv2.rotate(g, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                          270: cv2.ROTATE_90_COUNTERCLOCKWISE}[a])


class DS(tf.keras.utils.Sequence):
    """Feeds RAW [0,255] patches; EfficientNet carries its own normalisation."""

    def __init__(self, X, y, augment, shuffle):
        self.X, self.y = X, y.astype("float32")
        self.augment, self.shuffle = augment, shuffle
        self.idx = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / BATCH))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __getitem__(self, b):
        ids = self.idx[b * BATCH:(b + 1) * BATCH]
        out = np.empty((len(ids), PATCH, PATCH, 3), "float32")
        for j, i in enumerate(ids):
            g = self.X[i]
            if self.augment:
                g = rot(g, np.random.choice([0, 90, 180, 270]))
                if np.random.rand() < 0.5:
                    g = np.fliplr(g)
            out[j] = np.stack([g.astype("float32")] * 3, -1)
        return out, self.y[ids]


def as_batch(X):
    return np.stack([np.stack([g.astype("float32")] * 3, -1) for g in X])


def build():
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(PATCH, PATCH, 3))
    base.trainable = False
    inp = Input((PATCH, PATCH, 3))
    x = base(inp, training=False)
    x = Dropout(0.4)(GlobalAveragePooling2D()(x))
    return Model(inp, Dense(1, activation="sigmoid")(x)), base


def train_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    itr, iva = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(
        np.arange(tr_mask.sum()), cb.loc[tr_mask, "label"], groups=cb.loc[tr_mask, "group"]))
    Xtr_all, ytr_all = Xcb[tr_mask], cb.loc[tr_mask, "label"].values
    Xtr, ytr, Xva, yva = Xtr_all[itr], ytr_all[itr], Xtr_all[iva], ytr_all[iva]
    cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}

    m, base = build()
    e1, e2 = (2, 2) if args.quick else (5, 20)
    m.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    m.fit(DS(Xtr, ytr, True, True), validation_data=DS(Xva, yva, False, False),
          epochs=e1, class_weight=cw, verbose=0)
    base.trainable = True
    for l in base.layers[:-40]:
        l.trainable = False
    m.compile(tf.keras.optimizers.Adam(1e-5), "binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    m.fit(DS(Xtr, ytr, True, True), validation_data=DS(Xva, yva, False, False),
          epochs=e2, class_weight=cw, verbose=0,
          callbacks=[EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                   restore_best_weights=True)])
    return m, iva


rows = []
summary = []
csv_path = os.path.join(OUT, "predictions_main.csv")

for si, seed in enumerate(SEEDS):
    ts = time.time()
    model, iva = train_seed(seed)

    # The validation split is source-domain held-out data. It is the only set a
    # deployed model could legitimately calibrate on, so its scores are kept for
    # the threshold-transfer experiment.
    cb_tr = cb[tr_mask].reset_index(drop=True)
    va_rows = cb_tr.iloc[iva]
    packs = [
        ("cbis_val", as_batch(Xcb[tr_mask][iva]), va_rows["label"].values,
         va_rows["sample_id"].values, va_rows["group"].values),
        ("cbis_test", as_batch(Xcb[te_mask]), cb.loc[te_mask, "label"].values,
         cb.loc[te_mask, "sample_id"].values, cb.loc[te_mask, "group"].values),
        ("mias", as_batch(Xmi), mi["label"].values, mi["sample_id"].values, mi["group"].values),
        ("inbreast", as_batch(Xinb), inb["label"].values, inb["sample_id"].values, inb["group"].values),
    ]
    line = {"seed": seed}
    for name, X, y, sid, grp in packs:
        p = model.predict(X, batch_size=BATCH, verbose=0).ravel()
        rows.append(pd.DataFrame({"seed": seed, "dataset": name, "sample_id": sid,
                                  "group": grp, "label": y, "prob": p}))
        line[name] = roc_auc_score(y, p)
    summary.append(line)
    pd.concat(rows, ignore_index=True).to_csv(csv_path, index=False)
    model.save_weights(os.path.join(WEIGHTS, f"effb0_seed{seed}.weights.h5"))
    log(f"seed {seed} ({si + 1}/{len(SEEDS)}, {time.time() - ts:.0f}s): "
        + " | ".join(f"{k} {v:.4f}" for k, v in line.items() if k != "seed"))
    tf.keras.backend.clear_session()

S = pd.DataFrame(summary)
S.to_csv(os.path.join(OUT, "seed_summary_main.csv"), index=False)
log("=== AUC across seeds ===")
for c in ["cbis_val", "cbis_test", "mias", "inbreast"]:
    log(f"  {c:10s} mean {S[c].mean():.4f}  std {S[c].std(ddof=0):.4f}  "
        f"min {S[c].min():.4f}  max {S[c].max():.4f}")

with open(os.path.join(OUT, "environment_main.json"), "w") as f:
    json.dump({"tensorflow": tf.__version__, "numpy": np.__version__,
               "pandas": pd.__version__, "opencv": cv2.__version__,
               "python": platform.python_version(), "platform": platform.platform(),
               "gpus": [g.name for g in tf.config.list_physical_devices("GPU")],
               "seeds": SEEDS, "patch": PATCH, "batch": BATCH,
               "phase1": {"epochs": 5, "lr": 1e-3, "trainable": "head only"},
               "phase2": {"epochs": 20, "lr": 1e-5, "unfrozen_layers": 40,
                          "early_stopping": "val_auc, patience 5, restore best"}},
              f, indent=2)
log(f"wrote {csv_path} ({sum(len(r) for r in rows)} predictions)")
log("DONE")

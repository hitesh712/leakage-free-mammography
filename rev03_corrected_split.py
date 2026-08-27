"""Re-run the headline model under a strictly patient-disjoint CBIS-DDSM split.

The split audit (rev02) showed that the CBIS-DDSM *official* train/test
partition is not patient-disjoint once mass and calc cases are pooled: the two
partitions were drawn independently, so 31 patients contribute a mass case to
one side and a calc case to the other. 60 of the 704 official test lesions
(8.5%) belong to such patients, and in 19 of the 31 the same breast and view
appears on both sides.

This script quantifies what that costs by retraining the identical model on a
repaired, strictly patient-disjoint training set and comparing against the
official-split run from rev01.

Usage:  python revision/rev03_corrected_split.py [--seeds N] [--quick]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import revlib

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "out")
WEIGHTS = os.path.join(HERE, "weights")
os.makedirs(WEIGHTS, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
SEEDS = revlib.SEEDS[: args.seeds]

cb = pd.read_csv(os.path.join(CACHE, "cbis_manifest.csv"))
Xcb = np.load(os.path.join(CACHE, "cbis_roi_224.npy"))
mi = pd.read_csv(os.path.join(CACHE, "mias_manifest.csv"))
Xmi = np.load(os.path.join(CACHE, "mias_roi_224.npy"))
inb = pd.read_csv(os.path.join(CACHE, "inbreast_manifest.csv"))
Xinb = np.load(os.path.join(CACHE, "inbreast_roi_224.npy"))

cb, n_dropped, clash_patients = revlib.patient_disjoint_split(cb)
cb.to_csv(os.path.join(CACHE, "cbis_manifest_pd.csv"), index=False)
log(f"patient-disjoint repair: {len(clash_patients)} patients straddle the official split; "
    f"dropped {n_dropped} training lesions")
log(f"  train {int((cb.split_pd == 'train').sum())} (was {int((cb.split == 'train').sum())}) | "
    f"test {int((cb.split_pd == 'test').sum())}")

tr_mask = (cb["split_pd"] == "train").values
te_mask = (cb["split_pd"] == "test").values

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

log(f"TF {tf.__version__} | GPUs {[g.name for g in tf.config.list_physical_devices('GPU')]}")

rows, summary = [], []
csv_path = os.path.join(OUT, "predictions_corrected.csv")
Xtr_all, ytr_all = Xcb[tr_mask], cb.loc[tr_mask, "label"].values
cb_tr = cb[tr_mask].reset_index(drop=True)
epochs = (2, 2) if args.quick else (5, 20)

for si, seed in enumerate(SEEDS):
    ts = time.time()
    itr, iva = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(
        np.arange(len(cb_tr)), cb_tr["label"], groups=cb_tr["group"]))
    model = revlib.train_effb0(tf, Xtr_all[itr], ytr_all[itr], Xtr_all[iva], ytr_all[iva],
                               seed, epochs=epochs)
    va_rows = cb_tr.iloc[iva]
    packs = [
        ("cbis_val", revlib.as_batch(Xtr_all[iva]), va_rows["label"].values,
         va_rows["sample_id"].values, va_rows["group"].values),
        ("cbis_test", revlib.as_batch(Xcb[te_mask]), cb.loc[te_mask, "label"].values,
         cb.loc[te_mask, "sample_id"].values, cb.loc[te_mask, "group"].values),
        ("mias", revlib.as_batch(Xmi), mi["label"].values, mi["sample_id"].values, mi["group"].values),
        ("inbreast", revlib.as_batch(Xinb), inb["label"].values, inb["sample_id"].values,
         inb["group"].values),
    ]
    line = {"seed": seed}
    for name, X, y, sid, grp in packs:
        p = model.predict(X, batch_size=revlib.BATCH, verbose=0).ravel()
        rows.append(pd.DataFrame({"seed": seed, "dataset": name, "sample_id": sid,
                                  "group": grp, "label": y, "prob": p}))
        line[name] = roc_auc_score(y, p)
    summary.append(line)
    pd.concat(rows, ignore_index=True).to_csv(csv_path, index=False)
    model.save_weights(os.path.join(WEIGHTS, f"effb0_pd_seed{seed}.weights.h5"))
    log(f"seed {seed} ({si + 1}/{len(SEEDS)}, {time.time() - ts:.0f}s): "
        + " | ".join(f"{k} {v:.4f}" for k, v in line.items() if k != "seed"))
    tf.keras.backend.clear_session()

S = pd.DataFrame(summary)
S.to_csv(os.path.join(OUT, "seed_summary_corrected.csv"), index=False)
log("=== AUC across seeds (patient-disjoint split) ===")
for c in ["cbis_val", "cbis_test", "mias", "inbreast"]:
    log(f"  {c:10s} mean {S[c].mean():.4f}  std {S[c].std(ddof=0):.4f}")

with open(os.path.join(OUT, "corrected_split_info.json"), "w") as f:
    json.dump({"clash_patients": clash_patients, "n_train_lesions_dropped": n_dropped,
               "train_after": int(tr_mask.sum()), "test": int(te_mask.sum())}, f, indent=2)
log("DONE")

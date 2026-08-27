"""Exact split composition and a machine-checked patient-independence audit.

Reviewer 2, comment 1: "Report the exact numbers of patients, images, lesions and
patches in each data split, and confirm complete patient-level independence."

Emits:
  out/split_counts.csv        one row per dataset/split
  out/split_audit.json        overlap checks, printed verbatim in the paper
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

cb = pd.read_csv(os.path.join(CACHE, "cbis_manifest.csv"))
cbw = pd.read_csv(os.path.join(CACHE, "cbis_whole_manifest.csv"))
mi = pd.read_csv(os.path.join(CACHE, "mias_manifest.csv"))
inb = pd.read_csv(os.path.join(CACHE, "inbreast_manifest.csv"))

rows, audit = [], {}

# --------------------------------------------------------------------------
# CBIS-DDSM: patients / full images / lesion ROIs, per official split
# --------------------------------------------------------------------------
# The lesion manifest carries the parent full-mammogram path, so "images" is
# well defined even though the model consumes crops.
for split in ("train", "test"):
    s = cb[cb["split"] == split]
    rows.append({
        "dataset": "CBIS-DDSM", "split": split,
        "patients": s["group"].nunique(),
        "images": s["full"].nunique(),
        "lesions": len(s),
        "patches": len(s),
        "benign": int((s["label"] == 0).sum()),
        "malignant": int((s["label"] == 1).sum()),
    })

# The validation carve-out is patient-wise and reseeded per run; report its
# typical size across the ten seeds actually used.
tr = cb[cb["split"] == "train"].reset_index(drop=True)
va_sizes, va_pats = [], []
for seed in [2021, 7, 123, 42, 2024, 5, 777, 31337, 99, 1234]:
    _, iva = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(
        tr, tr["label"], groups=tr["group"]))
    va_sizes.append(len(iva))
    va_pats.append(tr.iloc[iva]["group"].nunique())
audit["cbis_val_carveout"] = {
    "rule": "GroupShuffleSplit(test_size=0.15) on patient_id, reseeded per run",
    "patches_min": int(min(va_sizes)), "patches_max": int(max(va_sizes)),
    "patches_mean": float(np.mean(va_sizes)),
    "patients_min": int(min(va_pats)), "patients_max": int(max(va_pats)),
}

tr_p = set(cb[cb["split"] == "train"]["group"])
te_p = set(cb[cb["split"] == "test"]["group"])
audit["cbis_train_test_patient_overlap"] = sorted(tr_p & te_p)
audit["cbis_patients_total"] = cb["group"].nunique()

# Rule out the stronger failure mode: the same image, or the same pixels,
# appearing on both sides of the split.
tr_rows, te_rows = cb[cb["split"] == "train"], cb[cb["split"] == "test"]
audit["cbis_shared_crop_files"] = len(set(tr_rows["img"]) & set(te_rows["img"]))
audit["cbis_shared_full_mammograms"] = len(
    set(tr_rows["full"].dropna()) & set(te_rows["full"].dropna()))
X = np.load(os.path.join(CACHE, "cbis_roi_224.npy"))
trm, tem = (cb["split"] == "train").values, (cb["split"] == "test").values
h_tr = {hash(X[i].tobytes()) for i in np.where(trm)[0]}
audit["cbis_test_patches_with_pixels_seen_in_train"] = int(
    sum(hash(X[i].tobytes()) in h_tr for i in np.where(tem)[0]))

# Where patients straddle the split, is it also the same breast and view?
# Breast/view live in the raw case descriptions rather than the patch manifest.
raw = []
for kind in ("mass", "calc"):
    for sp in ("train", "test"):
        fs = glob.glob(f"D:/cbis-ddsm/csv/{kind}_case_description_{sp}_set.csv")
        if fs:
            d = pd.read_csv(fs[0])
            d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
            d["split"] = sp
            d["kind"] = kind
            raw.append(d)
raw = pd.concat(raw, ignore_index=True)

# Each subset is internally patient-disjoint; the clash is only across subsets.
audit["cbis_overlap_within_mass"] = len(
    set(raw[(raw.kind == "mass") & (raw.split == "train")].patient_id)
    & set(raw[(raw.kind == "mass") & (raw.split == "test")].patient_id))
audit["cbis_overlap_within_calc"] = len(
    set(raw[(raw.kind == "calc") & (raw.split == "train")].patient_id)
    & set(raw[(raw.kind == "calc") & (raw.split == "test")].patient_id))

straddling = set(raw[raw.split == "train"].patient_id) & set(raw[raw.split == "test"].patient_id)
same_view = 0
for p in straddling:
    a = raw[(raw.patient_id == p) & (raw.split == "train")]
    b = raw[(raw.patient_id == p) & (raw.split == "test")]
    if set(zip(a.left_or_right_breast, a.image_view)) & set(zip(b.left_or_right_breast, b.image_view)):
        same_view += 1
te_raw = raw[raw.split == "test"]
audit["cbis_straddling_patients"] = len(straddling)
audit["cbis_straddling_patients_sharing_breast_and_view"] = same_view
audit["cbis_test_lesions_from_straddling_patients"] = int(
    te_raw.patient_id.isin(straddling).sum())
audit["cbis_test_lesions_total"] = int(len(te_raw))

# A patient must never straddle the train/validation boundary either.
worst = 0
for seed in [2021, 7, 123, 42, 2024, 5, 777, 31337, 99, 1234]:
    itr, iva = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(
        tr, tr["label"], groups=tr["group"]))
    worst = max(worst, len(set(tr.iloc[itr]["group"]) & set(tr.iloc[iva]["group"])))
audit["cbis_train_val_patient_overlap_worst_over_10_seeds"] = worst

# --------------------------------------------------------------------------
# MIAS: films are supplied as left/right pairs, so patient = ceil(refnum / 2)
# --------------------------------------------------------------------------
mi["patient"] = mi["refnum"].map(lambda r: (int(re.sub(r"\D", "", r)) + 1) // 2)
rows.append({
    "dataset": "Mini-MIAS", "split": "external test",
    "patients": mi["patient"].nunique(), "images": mi["refnum"].nunique(),
    "lesions": len(mi), "patches": len(mi),
    "benign": int((mi["label"] == 0).sum()), "malignant": int((mi["label"] == 1).sum()),
})
audit["mias_patient_rule"] = "MIAS films are released as L/R pairs; patient = ceil(refnum/2)"

# --------------------------------------------------------------------------
# INbreast: the public release redacts Patient ID, so try to recover a patient
# grouping from the anonymisation token embedded in the DICOM filenames.
# --------------------------------------------------------------------------
tok = {}
for p in glob.glob("D:/INbreast Release 1.0/AllDICOMs/*.dcm"):
    parts = os.path.basename(p).split("_")
    if len(parts) >= 2:
        tok[parts[0]] = parts[1]
inb["patient_token"] = inb["refnum"].astype(str).map(tok)
n_tok = inb["patient_token"].nunique(dropna=True)
audit["inbreast_patient_id_in_csv"] = "redacted ('removed') in the public release"
audit["inbreast_recovered_patient_tokens"] = int(n_tok)
audit["inbreast_images"] = int(inb["refnum"].nunique())
rows.append({
    "dataset": "INbreast", "split": "external test",
    "patients": int(n_tok), "images": inb["refnum"].nunique(),
    "lesions": len(inb), "patches": len(inb),
    "benign": int((inb["label"] == 0).sum()), "malignant": int((inb["label"] == 1).sum()),
})

# BI-RADS composition of the INbreast "benign" class -- reviewer 5, comment 2.
bd = inb["birads"].astype(str).value_counts().to_dict()
ben = inb[inb["label"] == 0]["birads"].astype(str).value_counts().to_dict()
audit["inbreast_birads_all"] = bd
audit["inbreast_birads_benign_class"] = ben
audit["inbreast_birads3_share_of_benign"] = round(
    100.0 * ben.get("3", 0) / max(1, sum(ben.values())), 1)

# --------------------------------------------------------------------------
# Whole-image CBIS baseline set
# --------------------------------------------------------------------------
lesions_per_full = cb.groupby("full").size()
cbw["n_lesions"] = cbw["full"].map(lesions_per_full).fillna(0).astype(int)
for split in ("train", "test"):
    s = cbw[cbw["split"] == split]
    rows.append({
        "dataset": "CBIS-DDSM (whole image)", "split": split,
        "patients": s["group"].nunique(), "images": len(s),
        "lesions": int(s["n_lesions"].sum()), "patches": len(s),
        "benign": int((s["label"] == 0).sum()), "malignant": int((s["label"] == 1).sum()),
    })
audit["cbis_whole_train_test_patient_overlap"] = sorted(
    set(cbw[cbw.split == "train"]["group"]) & set(cbw[cbw.split == "test"]["group"]))

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "split_counts.csv"), index=False)
with open(os.path.join(OUT, "split_audit.json"), "w") as f:
    json.dump(audit, f, indent=2)

print(T.to_string(index=False))
print()
print(json.dumps(audit, indent=2))

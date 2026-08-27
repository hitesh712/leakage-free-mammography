"""Build and cache every patch set used by the JJCIT revision experiments.

Preprocessing is byte-for-byte identical to the original submission scripts
(run_stats_rigor.py / run_extval_mias.py / run_extval_inbreast.py) so that the
re-run reproduces the published numbers:

    grayscale -> resize(PATCH, PATCH) -> CLAHE(clip=2.0, tiles=8x8)

New in the revision: every sample carries a group identifier (patient for
CBIS-DDSM, source image for MIAS/INbreast) so that bootstrap confidence
intervals can be computed at a stated level, and lesion geometry is retained so
that crop perturbation and Grad-CAM localisation can be evaluated later.

Usage:  python revision/prep_data.py [--sizes 224 300] [--whole]
"""
import argparse
import glob
import os
import plistlib
import sys
import time

import cv2
import numpy as np
import pandas as pd

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

CBIS = os.environ.get("CBIS_ROOT", "D:/cbis-ddsm")
MIAS_DIR = "D:/all-mias/"
INB = "D:/INbreast Release 1.0"

_clahe = cv2.createCLAHE(2.0, (8, 8))


def prep(gray, size):
    """The one and only preprocessing path: resize then CLAHE."""
    return _clahe.apply(cv2.resize(gray, (size, size)))


# --------------------------------------------------------------------------
# CBIS-DDSM
# --------------------------------------------------------------------------
def cbis_manifest():
    csv_dir = next(
        (c for c in [CBIS + "/csv", CBIS, CBIS + "/CBIS-DDSM/csv"]
         if os.path.isdir(c) and glob.glob(c + "/*case_description*")),
        None,
    )
    if csv_dir is None:
        sys.exit(f"ERROR: no CBIS case-description CSVs under {CBIS}")

    def load(kind, split):
        f = glob.glob(f"{csv_dir}/{kind}_case_description_{split}_set.csv")
        if not f:
            return None
        d = pd.read_csv(f[0])
        d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
        d["kind"] = kind
        d["split"] = split
        return d

    cases = pd.concat(
        [x for x in (load(k, s) for k in ("mass", "calc") for s in ("train", "test")) if x is not None],
        ignore_index=True,
    )
    cases["label"] = cases["pathology"].str.upper().map(lambda v: 1 if "MALIGNANT" in str(v) else 0)

    roi_col = next((c for c in cases.columns if "cropped" in c and "path" in c), None) \
        or next((c for c in cases.columns if "roi" in c and "path" in c), None)
    mask_col = next((c for c in cases.columns if "roi" in c and "mask" in c and "path" in c), None)
    # The full mammogram lives in `image_file_path` -- there is no column named "full".
    full_col = "image_file_path" if "image_file_path" in cases.columns else None

    di = pd.read_csv(glob.glob(f"{csv_dir}/dicom_info.csv")[0])
    di.columns = [c.strip().lower().replace(" ", "_") for c in di.columns]
    pathcol = "image_path" if "image_path" in di.columns else next(
        (c for c in di.columns if c in ("file_path", "path")), None)
    uidcol = next((c for c in di.columns if "seriesinstanceuid" in c or c == "series_uid"), None)

    # A crop and its ROI mask frequently share one SeriesInstanceUID (they are two
    # files inside the same series), so a flat uid -> path dict silently loses one
    # of them. Key by (uid, series description) instead.
    uid2paths = {}
    for _, r in di.iterrows():
        p = str(r[pathcol]).replace("CBIS-DDSM/", "").replace("\\", "/")
        full = os.path.join(CBIS, p)
        uid2paths.setdefault(str(r[uidcol]), {})[str(r.get("seriesdescription", ""))] = \
            full if os.path.exists(full) else p

    def resolve(cell, want=None):
        for uid in reversed([p for p in str(cell).replace("\\", "/").split("/") if p]):
            by_desc = uid2paths.get(uid)
            if not by_desc:
                continue
            for desc, path in by_desc.items():
                if want is not None and want not in desc:
                    continue
                if os.path.exists(path):
                    return path
        return None

    # Prefer the series tagged "cropped images"; fall back to any resolvable series
    # so the lesion set stays identical to the originally submitted 3,567 ROIs.
    cases["img"] = cases[roi_col].map(lambda c: resolve(c, "cropped") or resolve(c))
    cases["mask"] = cases[mask_col].map(lambda c: resolve(c, "ROI mask")) if mask_col else None
    cases["full"] = cases[full_col].map(lambda c: resolve(c, "full mammogram")) if full_col else None
    cases = cases[cases["img"].notna()].reset_index(drop=True)
    cases["group"] = cases["patient_id"].astype(str)
    cases["sample_id"] = [f"cbis_{i:05d}" for i in range(len(cases))]
    return cases


def build_cbis(sizes):
    cases = cbis_manifest()
    log(f"CBIS resolved {len(cases)} lesion ROIs | "
        f"train {(cases.split == 'train').sum()} test {(cases.split == 'test').sum()} | "
        f"labels {cases['label'].value_counts().to_dict()}")

    keep = np.ones(len(cases), bool)
    stacks = {s: np.zeros((len(cases), s, s), np.uint8) for s in sizes}
    for i, path in enumerate(cases["img"].values):
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            keep[i] = False
            continue
        for s in sizes:
            stacks[s][i] = prep(g, s)
    cases = cases[keep].reset_index(drop=True)
    for s in sizes:
        np.save(os.path.join(CACHE, f"cbis_roi_{s}.npy"), stacks[s][keep])
        log(f"  cached cbis_roi_{s}.npy {stacks[s][keep].shape}")

    cols = ["sample_id", "group", "label", "split", "kind", "img", "mask", "full"]
    cases[cols].to_csv(os.path.join(CACHE, "cbis_manifest.csv"), index=False)
    log(f"  manifest -> cbis_manifest.csv ({len(cases)} rows, "
        f"{cases['group'].nunique()} patients, masks {cases['mask'].notna().sum()})")
    return cases


def build_cbis_whole(cases, size=224):
    """Whole-mammogram baseline set (reviewer 1, comment 7)."""
    # One row per full mammogram. An image counts as malignant if ANY lesion it
    # contains is malignant -- the standard whole-image formulation.
    have = cases[cases["full"].notna()].copy()
    agg = have.groupby("full").agg(label=("label", "max"), group=("group", "first"),
                                   split=("split", "first"), n_lesions=("label", "size"))
    sub = agg.reset_index()
    sub["sample_id"] = [f"cbisw_{i:05d}" for i in range(len(sub))]
    log(f"CBIS whole-image: {len(sub)} unique full mammograms "
        f"({sub['group'].nunique()} patients, labels {sub['label'].value_counts().to_dict()}, "
        f"train {(sub.split == 'train').sum()} test {(sub.split == 'test').sum()})")
    arr = np.zeros((len(sub), size, size), np.uint8)
    keep = np.ones(len(sub), bool)
    for i, path in enumerate(sub["full"].values):
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            keep[i] = False
            continue
        arr[i] = prep(g, size)
    sub = sub[keep].reset_index(drop=True)
    np.save(os.path.join(CACHE, f"cbis_whole_{size}.npy"), arr[keep])
    sub[["sample_id", "group", "label", "split", "full"]].to_csv(
        os.path.join(CACHE, "cbis_whole_manifest.csv"), index=False)
    log(f"  cached cbis_whole_{size}.npy {arr[keep].shape}")


# --------------------------------------------------------------------------
# MIAS
# --------------------------------------------------------------------------
def mias_lesions():
    rows = []
    for line in open(MIAS_DIR + "Info1.txt"):
        q = line.strip().split()
        if len(q) < 7 or not q[0].startswith("mdb") or q[3] not in ("B", "M"):
            continue
        try:
            x, y, r = int(q[4]), int(q[5]), int(q[6])
        except (IndexError, ValueError):
            continue
        rows.append({"refnum": q[0], "bg": q[2], "cls": q[1], "sev": q[3],
                     "x": x, "y": y, "r": r, "label": 0 if q[3] == "B" else 1})
    return pd.DataFrame(rows)


def mias_crop(img, x, y, r, size, shift=(0.0, 0.0), scale=1.0):
    """Lesion crop. MIAS uses a bottom-left origin, so the row is flipped."""
    H, W = img.shape
    cx, cy = x, H - y
    side = int(np.clip(2 * r * scale, 96, 450))
    half = side // 2
    cx = int(round(cx + shift[0] * side))
    cy = int(round(cy + shift[1] * side))
    cx = int(np.clip(cx, half, W - half))
    cy = int(np.clip(cy, half, H - half))
    return prep(img[cy - half:cy - half + side, cx - half:cx - half + side], size), (cx, cy, side)


def build_mias(sizes):
    L = mias_lesions()
    stacks = {s: [] for s in sizes}
    keep = []
    geo = []
    for i, row in L.iterrows():
        img = cv2.imread(MIAS_DIR + row["refnum"] + ".pgm", cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        keep.append(i)
        for s in sizes:
            patch, g = mias_crop(img, row["x"], row["y"], row["r"], s)
            stacks[s].append(patch)
        geo.append(g)
    L = L.loc[keep].reset_index(drop=True)
    L["cx"], L["cy"], L["side"] = [g[0] for g in geo], [g[1] for g in geo], [g[2] for g in geo]
    L["group"] = L["refnum"]
    L["sample_id"] = [f"mias_{i:04d}" for i in range(len(L))]
    for s in sizes:
        np.save(os.path.join(CACHE, f"mias_roi_{s}.npy"), np.array(stacks[s], np.uint8))
    L.to_csv(os.path.join(CACHE, "mias_manifest.csv"), index=False)
    log(f"MIAS {len(L)} lesion patches | {L['group'].nunique()} images | "
        f"labels {L['label'].value_counts().to_dict()} (0=B,1=M)")
    return L


# --------------------------------------------------------------------------
# INbreast
# --------------------------------------------------------------------------
def inbreast_boxes(xmlpath):
    try:
        with open(xmlpath, "rb") as f:
            pl = plistlib.load(f)
    except Exception:
        return []
    boxes = []
    for img in pl.get("Images", []):
        for roi in img.get("ROIs", []):
            name = str(roi.get("Name", "")).lower()
            if "mass" not in name and "spicul" not in name:
                continue
            pts = []
            for s in roi.get("Point_px", []):
                try:
                    x, y = s.strip("()").split(",")
                    pts.append((float(x), float(y)))
                except Exception:
                    pass
            if len(pts) >= 3:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def read_dcm(path):
    import pydicom
    a = pydicom.dcmread(path).pixel_array.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def inbreast_crop(img, bbox, size, shift=(0.0, 0.0), scale=1.0):
    H, W = img.shape
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(np.clip(max(x1 - x0, y1 - y0) * 1.3 * scale, 96, 800))
    half = side // 2
    cx = int(np.clip(round(cx + shift[0] * side), half, W - half))
    cy = int(np.clip(round(cy + shift[1] * side), half, H - half))
    return prep(img[cy - half:cy - half + side, cx - half:cx - half + side], size)


def build_inbreast(sizes):
    csv = pd.read_csv(INB + "/INbreast.csv", sep=";", dtype=str)
    csv.columns = [c.strip() for c in csv.columns]

    def b2l(b):
        b = str(b).strip().lower()
        if not b or b == "nan":
            return None
        return 0 if int("".join(ch for ch in b if ch.isdigit())[:1]) <= 3 else 1

    fncol = next(c for c in csv.columns if c.lower().replace(" ", "") == "filename")
    id2label = {str(r[fncol]).strip(): b2l(r["Bi-Rads"]) for _, r in csv.iterrows()}
    id2birads = {str(r[fncol]).strip(): str(r["Bi-Rads"]).strip() for _, r in csv.iterrows()}
    id2dcm = {os.path.basename(p).split("_")[0]: p for p in glob.glob(INB + "/AllDICOMs/*.dcm")}

    rows, stacks = [], {s: [] for s in sizes}
    for xp in sorted(glob.glob(INB + "/AllXML/*.xml")):
        iid = os.path.splitext(os.path.basename(xp))[0]
        lab = id2label.get(iid)
        if lab is None or iid not in id2dcm:
            continue
        boxes = inbreast_boxes(xp)
        if not boxes:
            continue
        img = read_dcm(id2dcm[iid])
        for bb in boxes:
            for s in sizes:
                stacks[s].append(inbreast_crop(img, bb, s))
            rows.append({"refnum": iid, "label": lab, "birads": id2birads.get(iid, ""),
                         "x0": bb[0], "y0": bb[1], "x1": bb[2], "y1": bb[3],
                         "dcm": id2dcm[iid]})
    I = pd.DataFrame(rows)
    I["group"] = I["refnum"]
    I["sample_id"] = [f"inb_{i:04d}" for i in range(len(I))]
    for s in sizes:
        np.save(os.path.join(CACHE, f"inbreast_roi_{s}.npy"), np.array(stacks[s], np.uint8))
    I.to_csv(os.path.join(CACHE, "inbreast_manifest.csv"), index=False)
    log(f"INbreast {len(I)} mass patches | {I['group'].nunique()} images | "
        f"labels {I['label'].value_counts().to_dict()} | BI-RADS {I['birads'].value_counts().to_dict()}")
    return I


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 300])
    ap.add_argument("--whole", action="store_true", help="also build the whole-mammogram baseline set")
    a = ap.parse_args()

    cases = build_cbis(a.sizes)
    if a.whole:
        build_cbis_whole(cases, size=224)
    build_mias(a.sizes)
    build_inbreast(a.sizes)
    log("DONE")

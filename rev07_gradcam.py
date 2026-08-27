"""Quantitative Grad-CAM localisation, with success and failure examples.

Reviewer 1, comment 8 and reviewer 5, minor 3: the Grad-CAM panels are
qualitative, so the manuscript may only claim the maps *suggest* lesion-driven
prediction. This measures it.

Design note. If every patch is cropped centred on its lesion, then "the peak
activation falls inside the ROI" is satisfied by any model with a centre bias
and the metric is vacuous. Each crop here is therefore deliberately
off-centred by a random 10-25% of the patch side, and the annotation mask is
cropped through the identical transform, so the lesion sits in a known but
non-central position. Two null models are scored on the same patches:

  centre prior  -- a fixed Gaussian blob at the patch centre
  random        -- a smoothed uniform-noise map

Metrics per patch:
  pointing hit  -- does the CAM argmax fall inside the lesion mask?
  energy ratio  -- fraction of total CAM mass inside the mask
  IoU@top20     -- IoU of the top-20% CAM pixels with the mask

Usage:  python revision/rev07_gradcam.py [--seeds 3] [--n 400]
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

import revlib
from prep_data import prep

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "out")
WEIGHTS = os.path.join(HERE, "weights")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)
MIAS_DIR = "D:/all-mias/"
PATCH = 224

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n", type=int, default=400, help="max CBIS test lesions to score")
args = ap.parse_args()
SEEDS = revlib.SEEDS[: args.seeds]

cb = pd.read_csv(os.path.join(CACHE, "cbis_manifest.csv"))
cbte = cb[(cb["split"] == "test") & cb["mask"].notna() & cb["full"].notna()].reset_index(drop=True)
mi = pd.read_csv(os.path.join(CACHE, "mias_manifest.csv"))

import tensorflow as tf

log(f"TF {tf.__version__} | GPUs {[g.name for g in tf.config.list_physical_devices('GPU')]}")

models, bases = [], []
for s in SEEDS:
    m, b = revlib.load_effb0(tf, os.path.join(WEIGHTS, f"effb0_seed{s}.weights.h5"))
    models.append(m)
    bases.append(b)
log(f"loaded {len(models)} models")

FEAT = [tf.keras.Model(b.input, b.get_layer("top_activation").output) for b in bases]
DENSE = [m.layers[-1] for m in models]


def gradcam(i, x):
    """Grad-CAM for model i on a single RAW [0,255] patch, returned at 224x224."""
    xb = tf.convert_to_tensor(x[None].astype("float32"))
    with tf.GradientTape() as tape:
        feats = FEAT[i](xb, training=False)
        tape.watch(feats)
        pooled = tf.reduce_mean(feats, axis=[1, 2])
        score = DENSE[i](pooled)
    grads = tape.gradient(score, feats)[0].numpy()
    fmap = feats[0].numpy()
    w = grads.mean(axis=(0, 1))
    cam = np.maximum((fmap * w).sum(-1), 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cv2.resize(cam, (PATCH, PATCH)), float(tf.sigmoid(score)[0, 0])


def metrics(cam, mask):
    if mask.sum() == 0:
        return None
    peak = np.unravel_index(np.argmax(cam), cam.shape)
    hit = bool(mask[peak])
    energy = float((cam * mask).sum() / max(cam.sum(), 1e-9))
    thr = np.quantile(cam, 0.80)
    top = cam >= thr
    iou = float((top & mask).sum() / max((top | mask).sum(), 1))
    return {"hit": hit, "energy": energy, "iou": iou}


_yy, _xx = np.mgrid[0:PATCH, 0:PATCH]
CENTRE_PRIOR = np.exp(-(((_xx - PATCH / 2) ** 2 + (_yy - PATCH / 2) ** 2) / (2 * (PATCH / 5) ** 2)))
_rng = np.random.default_rng(0)


def random_map():
    return cv2.GaussianBlur(_rng.random((PATCH, PATCH)).astype("float32"), (0, 0), 15)


# --------------------------------------------------------------------------
# build off-centred CBIS patches together with their annotation masks
# --------------------------------------------------------------------------
def build_cbis_samples(limit, context=1.6, shift_lo=0.10, shift_hi=0.25):
    rng = np.random.default_rng(1)
    out = []
    for _, r in cbte.iterrows():
        if len(out) >= limit:
            break
        mask_img = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
        full = cv2.imread(r["full"], cv2.IMREAD_GRAYSCALE)
        if mask_img is None or full is None:
            continue
        ys, xs = np.where(mask_img > 127)
        if len(xs) == 0:
            continue
        H, W = full.shape
        mh, mw = mask_img.shape
        sx, sy = W / mw, H / mh
        x0, y0, x1, y1 = xs.min() * sx, ys.min() * sy, xs.max() * sx, ys.max() * sy
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = int(np.clip(max(x1 - x0, y1 - y0) * context, 48, min(H, W)))
        half = side // 2
        # deliberate off-centring so the metric cannot be won by a centre bias
        th = rng.uniform(0, 2 * np.pi)
        frac = rng.uniform(shift_lo, shift_hi)
        cxs = int(np.clip(round(cx + frac * side * np.cos(th)), half, W - half))
        cys = int(np.clip(round(cy + frac * side * np.sin(th)), half, H - half))
        sl = (slice(cys - half, cys - half + side), slice(cxs - half, cxs - half + side))
        patch = prep(full[sl], PATCH)
        # crop the mask through the identical transform (mask is in its own scale)
        m_small = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)[sl]
        mres = cv2.resize(m_small, (PATCH, PATCH), interpolation=cv2.INTER_NEAREST) > 127
        if mres.sum() < 20:
            continue
        out.append({"patch": patch, "mask": mres, "label": int(r["label"]),
                    "sample_id": r["sample_id"]})
    return out


def build_mias_samples(context=1.0, shift_lo=0.10, shift_hi=0.25):
    rng = np.random.default_rng(2)
    out = []
    for _, r in mi.iterrows():
        img = cv2.imread(MIAS_DIR + r["refnum"] + ".pgm", cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        cx, cy = r["x"], H - r["y"]
        side = int(np.clip(2 * r["r"] * context, 96, 450))
        half = side // 2
        th = rng.uniform(0, 2 * np.pi)
        frac = rng.uniform(shift_lo, shift_hi)
        cxs = int(np.clip(round(cx + frac * side * np.cos(th)), half, W - half))
        cys = int(np.clip(round(cy + frac * side * np.sin(th)), half, H - half))
        sl = (slice(cys - half, cys - half + side), slice(cxs - half, cxs - half + side))
        patch = prep(img[sl], PATCH)
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (int(cx), int(cy)), int(r["r"]), 255, -1)
        mres = cv2.resize(m[sl], (PATCH, PATCH), interpolation=cv2.INTER_NEAREST) > 127
        if mres.sum() < 20:
            continue
        out.append({"patch": patch, "mask": mres, "label": int(r["label"]),
                    "sample_id": r["sample_id"]})
    return out


# Two cropping regimes.
#
#   native     the convention the model was trained on. The lesion fills a large
#              fraction of the patch, so a fixed centre blob is an extremely
#              strong baseline and the pointing game is close to uninformative.
#              Reported because it is the regime the model actually operates in.
#
#   wide field a 3x-context crop with larger off-centring, so the lesion covers
#              roughly a tenth of the patch. Here a centre prior is weak and the
#              metric genuinely tests whether saliency finds the lesion.
REGIMES = [
    ("native", dict(context=1.6), dict(context=1.0)),
    ("widefield", dict(context=3.0, shift_lo=0.20, shift_hi=0.33),
     dict(context=2.2, shift_lo=0.20, shift_hi=0.33)),
]

rows = []
store = {}
for regime, cbis_kw, mias_kw in REGIMES:
    for dsname, samples in ((f"cbis_test|{regime}", build_cbis_samples(args.n, **cbis_kw)),
                            (f"mias|{regime}", build_mias_samples(**mias_kw))):
        log(f"{dsname}: scoring {len(samples)} patches")
        store[dsname] = samples
        for s in samples:
            x = np.stack([s["patch"].astype("float32")] * 3, -1)
            cams, probs = [], []
            for i in range(len(models)):
                c, p = gradcam(i, x)
                cams.append(c)
                probs.append(p)
            cam = np.mean(cams, axis=0)
            prob = float(np.mean(probs))
            mm = metrics(cam, s["mask"])
            if mm is None:
                continue
            mc = metrics(CENTRE_PRIOR, s["mask"])
            mr = metrics(random_map(), s["mask"])
            rows.append({"dataset": dsname, "sample_id": s["sample_id"], "label": s["label"],
                         "prob": prob, "correct": int((prob >= 0.5) == s["label"]),
                         "mask_frac": float(s["mask"].mean()),
                         "hit": mm["hit"], "energy": mm["energy"], "iou": mm["iou"],
                         "hit_centre": mc["hit"], "energy_centre": mc["energy"],
                         "hit_random": mr["hit"], "energy_random": mr["energy"]})
            s["cam"] = cam
            s["prob"] = prob
            s["hit"] = mm["hit"]
            s["energy"] = mm["energy"]

G = pd.DataFrame(rows)
G.to_csv(os.path.join(OUT, "gradcam_localisation.csv"), index=False)

summary = []
for ds, g in G.groupby("dataset"):
    mf = g["mask_frac"].mean()
    # Under a uniform saliency map the energy inside the mask equals the mask's
    # area fraction, so energy/mask_frac is a chance-normalised concentration:
    # 1.0 means no better than uniform, whatever the lesion size.
    summary.append({
        "dataset": ds, "n": len(g),
        "pointing_hit": g["hit"].mean(), "energy": g["energy"].mean(), "iou": g["iou"].mean(),
        "pointing_hit_centre": g["hit_centre"].mean(), "energy_centre": g["energy_centre"].mean(),
        "pointing_hit_random": g["hit_random"].mean(), "energy_random": g["energy_random"].mean(),
        "mask_frac": mf,
        "conc_gradcam": g["energy"].mean() / max(mf, 1e-9),
        "conc_centre": g["energy_centre"].mean() / max(mf, 1e-9),
        "conc_random": g["energy_random"].mean() / max(mf, 1e-9),
        "hit_when_correct": g[g.correct == 1]["hit"].mean(),
        "hit_when_wrong": g[g.correct == 0]["hit"].mean(),
    })
S = pd.DataFrame(summary)
S.to_csv(os.path.join(OUT, "gradcam_summary.csv"), index=False)
print(S.to_string(index=False))

# --------------------------------------------------------------------------
# figure: successes (top row) and failures (bottom row)
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for ds in store:
    ok = [s for s in store[ds] if s.get("hit") and (s["prob"] >= 0.5) == s["label"]]
    bad = [s for s in store[ds] if s.get("hit") is False]
    ok = sorted(ok, key=lambda s: -s["energy"])[:4]
    bad = sorted(bad, key=lambda s: s["energy"])[:4]
    if not ok or not bad:
        continue
    fig, ax = plt.subplots(2, 4, figsize=(11, 5.8), dpi=150)
    for col, s in enumerate(ok + bad):
        r = 0 if col < 4 else 1
        c = col % 4
        a = ax[r, c]
        a.imshow(s["patch"], cmap="gray")
        a.imshow(s["cam"], cmap="jet", alpha=0.42)
        cnt, _ = cv2.findContours(s["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        for cc in cnt:
            a.plot(cc[:, 0, 0], cc[:, 0, 1], "w-", lw=1.6)
        a.set_title(f"{'hit' if s['hit'] else 'miss'} | p={s['prob']:.2f} | "
                    f"{'M' if s['label'] else 'B'}", fontsize=9)
        a.axis("off")
    ax[0, 0].set_ylabel("success")
    dname, reg = ds.split("|")
    pretty = {"cbis_test": "CBIS-DDSM test", "mias": "Mini-MIAS"}[dname]
    regname = {"native": "native crop", "widefield": "wide-field crop"}[reg]
    fig.suptitle(f"Grad-CAM on {pretty} ({regname}): successes (top), failures (bottom).\n"
                 "White outline = radiologist annotation; crops are deliberately off-centred.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"gradcam_{ds.replace('|', '_')}.png"),
                dpi=250, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote figures/gradcam_{ds}.png")

log("DONE")

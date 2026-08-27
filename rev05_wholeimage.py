"""Whole-mammogram benign-vs-malignant baseline on CBIS-DDSM.

Reviewer 1, comment 7: the manuscript concludes that lesion patches outperform
whole-image analysis, but demonstrates it only on Mini-MIAS. This supplies the
missing whole-image baseline on the primary dataset, under the same
patient-wise leakage-free protocol as the patch experiments.

Run at two input sizes, because a full mammogram downsampled to 224 px may
simply destroy the lesion; 512 px tests whether the gap is a resolution
artefact or a genuine framing effect.

An image is labelled malignant if any lesion it contains is malignant.

Usage:  python revision/rev05_wholeimage.py [--sizes 224 512] [--seeds 3]
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "out")
DEV = "cuda"
_clahe = cv2.createCLAHE(2.0, (8, 8))

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[224, 512])
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
SEEDS = [2021, 7, 123, 42, 2024][: args.seeds]

W = pd.read_csv(os.path.join(CACHE, "cbis_whole_manifest.csv"))
log(f"whole mammograms {len(W)} | train {(W.split == 'train').sum()} test {(W.split == 'test').sum()} "
    f"| patients {W.group.nunique()} | labels {W.label.value_counts().to_dict()}")


def ensure_cache(size):
    path = os.path.join(CACHE, f"cbis_whole_{size}.npy")
    if os.path.exists(path):
        arr = np.load(path)
        if len(arr) == len(W):
            return arr
    log(f"building whole-image cache @{size} ({len(W)} images)...")
    arr = np.zeros((len(W), size, size), np.uint8)
    for i, p in enumerate(W["full"].values):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is not None:
            arr[i] = _clahe.apply(cv2.resize(g, (size, size)))
    np.save(path, arr)
    log(f"  cached {path}")
    return arr


def to_tensor(u8, mean, std):
    x = torch.from_numpy(u8).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - mean) / std


def augment(u8, rng):
    out = np.empty_like(u8)
    for i, g in enumerate(u8):
        # Whole mammograms have a fixed anatomical orientation, so only the
        # laterality flip is meaningful here -- no 90-degree rotations.
        out[i] = np.ascontiguousarray(np.fliplr(g) if rng.random() < 0.5 else g)
    return out


@torch.no_grad()
def predict(model, X, bs, mean, std):
    model.eval()
    ps = []
    for i in range(0, len(X), bs):
        xb = to_tensor(X[i:i + bs], mean, std).to(DEV)
        with torch.amp.autocast("cuda"):
            ps.append(torch.sigmoid(model(xb).float().squeeze(-1)).cpu().numpy())
    return np.concatenate(ps)


def set_phase(model, phase):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    params = list(model.parameters())
    cutoff = int(len(params) * 0.75)
    for i, p in enumerate(params):
        p.requires_grad = (id(p) in head_ids) if phase == 1 else (i >= cutoff or id(p) in head_ids)


def run(size, seed, X, bs):
    import timm
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1).to(DEV)
    dc = model.default_cfg
    mean = torch.tensor(dc["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(dc["std"]).view(1, 3, 1, 1)

    trm = (W["split"] == "train").values
    tem = (W["split"] == "test").values
    Wtr = W[trm].reset_index(drop=True)
    itr, iva = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(
        np.arange(len(Wtr)), Wtr["label"], groups=Wtr["group"]))
    Xtr, ytr = X[trm][itr], Wtr["label"].values[itr]
    Xva, yva = X[trm][iva], Wtr["label"].values[iva]
    Xte, yte = X[tem], W.loc[tem, "label"].values
    pw = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32).to(DEV)

    scaler = torch.amp.GradScaler("cuda")
    best = {"auc": -1, "state": None}
    e1, e2 = (1, 1) if args.quick else (5, 20)
    for phase, epochs, lr in ((1, e1, 1e-3), (2, e2, 1e-5)):
        set_phase(model, phase)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        patience = 0
        for ep in range(epochs):
            model.train()
            order = rng.permutation(len(Xtr))
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                xb = to_tensor(augment(Xtr[idx], rng), mean, std).to(DEV)
                yb = torch.from_numpy(ytr[idx]).float().to(DEV)
                with torch.amp.autocast("cuda"):
                    loss = F.binary_cross_entropy_with_logits(
                        model(xb).float().squeeze(-1), yb, pos_weight=pw)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            va = roc_auc_score(yva, predict(model, Xva, bs, mean, std))
            if phase == 2:
                if va > best["auc"]:
                    best = {"auc": va, "state": {k: v.detach().cpu().clone()
                                                 for k, v in model.state_dict().items()}}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 5:
                        break
    if best["state"]:
        model.load_state_dict(best["state"])
    p = predict(model, Xte, bs, mean, std)
    out = {"size": size, "seed": seed, "val_auc": best["auc"], "test_auc": roc_auc_score(yte, p)}
    del model
    torch.cuda.empty_cache()
    return out, p, yte


rows, preds = [], []
for size in args.sizes:
    X = ensure_cache(size)
    bs = 32 if size <= 224 else 8
    for seed in SEEDS:
        ts = time.time()
        r, p, yte = run(size, seed, X, bs)
        r["secs"] = round(time.time() - ts)
        rows.append(r)
        preds.append(pd.DataFrame({"size": size, "seed": seed,
                                   "sample_id": W.loc[W.split == "test", "sample_id"].values,
                                   "group": W.loc[W.split == "test", "group"].values,
                                   "label": yte, "prob": p}))
        log(f"  whole-image @{size} seed {seed} ({r['secs']}s): "
            f"val {r['val_auc']:.4f} | TEST {r['test_auc']:.4f}")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "wholeimage_runs.csv"), index=False)
        pd.concat(preds, ignore_index=True).to_csv(
            os.path.join(OUT, "predictions_wholeimage.csv"), index=False)

R = pd.DataFrame(rows)
agg = R.groupby("size").agg(test_auc_mean=("test_auc", "mean"),
                            test_auc_std=("test_auc", "std"), n=("seed", "count")).reset_index()
agg.to_csv(os.path.join(OUT, "wholeimage_summary.csv"), index=False)
log("=== CBIS-DDSM whole-image baseline ===")
print(agg.to_string(index=False))
log("DONE")

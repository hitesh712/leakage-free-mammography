"""Unified architecture x resolution ablation under the identical leakage-free protocol.

Answers three reviewer requests at once:

  R1.5  "both the backbone and input resolution are changed at the same time...
         a cleaner ablation would keep the backbone fixed while varying only the
         image size."   -> EfficientNet-B0 and -B3 are each run at 224 and 300,
         giving a 2x2 factorial that separates backbone from resolution.

  R1.11 "all three selected models belong to a similar efficiency-focused family
         ... concluding a task/data ceiling may be premature."
  R5.3  "include at least one modern, heavier architecture (e.g. a Vision
         Transformer or a deeper ResNet) under the exact same leakage-free
         protocol."   -> ResNet-50, DenseNet-121, ViT-B/16 and Swin-T are added.

Everything here runs in one PyTorch/timm codebase so the comparison is
internally consistent; EfficientNet-B0 @224 doubles as a cross-framework anchor
against the TensorFlow result reported in the main table.

Protocol, identical for every configuration:
  phase 1  head only, 5 epochs, Adam 1e-3
  phase 2  deepest 25% of parameter tensors unfrozen, <=20 epochs, Adam 1e-5,
           early stopping on validation AUC (patience 5, restore best)
  patient-wise 15% validation carve-out, balanced class weights, rot90/flip aug.

Usage:  python revision/rev04_arch_ablation.py [--seeds 3] [--only NAME]
"""
import argparse
import json
import os
import time

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
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"

# name, timm id, input size, batch size
CONFIGS = [
    ("MobileNetV2 @224", "mobilenetv2_100", 224, 32),
    ("EfficientNet-B0 @224", "efficientnet_b0", 224, 32),
    ("EfficientNet-B0 @300", "efficientnet_b0", 300, 24),
    ("EfficientNet-B3 @224", "efficientnet_b3", 224, 24),
    ("EfficientNet-B3 @300", "efficientnet_b3", 300, 16),
    ("ResNet-50 @224", "resnet50", 224, 32),
    ("ResNet-50 @300", "resnet50", 300, 16),
    ("DenseNet-121 @224", "densenet121", 224, 24),
    ("ViT-B/16 @224", "vit_base_patch16_224", 224, 16),
    ("Swin-T @224", "swin_tiny_patch4_window7_224", 224, 16),
    ("ConvNeXt-T @224", "convnext_tiny", 224, 16),
]

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--only", type=str, default=None)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
SEEDS = [2021, 7, 123, 42, 2024][: args.seeds]

cb = pd.read_csv(os.path.join(CACHE, "cbis_manifest.csv"))
mi = pd.read_csv(os.path.join(CACHE, "mias_manifest.csv"))
inb = pd.read_csv(os.path.join(CACHE, "inbreast_manifest.csv"))
BANK = {s: {"cbis": np.load(os.path.join(CACHE, f"cbis_roi_{s}.npy")),
            "mias": np.load(os.path.join(CACHE, f"mias_roi_{s}.npy")),
            "inbreast": np.load(os.path.join(CACHE, f"inbreast_roi_{s}.npy"))}
        for s in (224, 300)}
tr_mask = (cb["split"] == "train").values
te_mask = (cb["split"] == "test").values
log(f"CBIS train {tr_mask.sum()} test {te_mask.sum()} | MIAS {len(mi)} | INbreast {len(inb)}")


def to_tensor(batch_u8, mean, std):
    """uint8 HxW grayscale -> normalised 3-channel float tensor."""
    x = torch.from_numpy(batch_u8).float().div_(255.0)
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - mean) / std


def augment(batch_u8, rng):
    out = np.empty_like(batch_u8)
    for i, g in enumerate(batch_u8):
        k = rng.integers(0, 4)
        g = np.rot90(g, k)
        if rng.random() < 0.5:
            g = np.fliplr(g)
        out[i] = np.ascontiguousarray(g)
    return out


@torch.no_grad()
def predict(model, X, bs, mean, std):
    model.eval()
    ps = []
    for i in range(0, len(X), bs):
        xb = to_tensor(X[i:i + bs], mean, std).to(DEV, non_blocking=True)
        with torch.amp.autocast("cuda"):
            ps.append(torch.sigmoid(model(xb).float().squeeze(-1)).cpu().numpy())
    return np.concatenate(ps)


def set_phase(model, phase):
    """phase 1 = classifier only; phase 2 = deepest 25% of tensors + classifier.

    The classifier is located via timm's own accessor and matched by parameter
    identity, which works uniformly across CNN and transformer families where
    the head is variously named fc / classifier / head.
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    params = list(model.parameters())
    cutoff = int(len(params) * 0.75)
    for i, p in enumerate(params):
        p.requires_grad = (id(p) in head_ids) if phase == 1 else (i >= cutoff or id(p) in head_ids)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_mode(model):
    """Training mode, but with batch normalisation held in inference mode.

    The TensorFlow pipeline calls the backbone as `base(inp, training=False)`
    throughout, so its BN layers always use the ImageNet running statistics.
    Letting PyTorch update BN running statistics instead --- on batches of 16-32
    grayscale mammogram patches --- destabilises the low-learning-rate
    fine-tuning stage badly enough to cost roughly 0.12 AUC, which would make
    this table a comparison of BN handling rather than of architectures.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def run(cfg_name, timm_id, size, bs, seed):
    import timm
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ViT/Swin are only run at their native 224; CNNs accept any input size.
    model = timm.create_model(timm_id, pretrained=True, num_classes=1).to(DEV)
    dc = model.default_cfg
    mean = torch.tensor(dc["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(dc["std"]).view(1, 3, 1, 1)

    Xall, yall = BANK[size]["cbis"], cb["label"].values
    Xtr_all, ytr_all = Xall[tr_mask], yall[tr_mask]
    cb_tr = cb[tr_mask].reset_index(drop=True)
    itr, iva = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(
        np.arange(len(cb_tr)), cb_tr["label"], groups=cb_tr["group"]))
    Xtr, ytr, Xva, yva = Xtr_all[itr], ytr_all[itr], Xtr_all[iva], ytr_all[iva]
    pw = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32).to(DEV)

    scaler = torch.amp.GradScaler("cuda")
    best = {"auc": -1, "state": None}
    e1, e2 = (1, 1) if args.quick else (5, 20)

    for phase, epochs, lr in ((1, e1, 1e-3), (2, e2, 1e-5)):
        ntr = set_phase(model, phase)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        if phase == 2:
            log(f"    phase2 trainable params {ntr / 1e6:.1f}M")
        patience = 0
        for ep in range(epochs):
            train_mode(model)
            order = rng.permutation(len(Xtr))
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                xb = to_tensor(augment(Xtr[idx], rng), mean, std).to(DEV, non_blocking=True)
                yb = torch.from_numpy(ytr[idx]).float().to(DEV)
                with torch.amp.autocast("cuda"):
                    loss = F.binary_cross_entropy_with_logits(
                        model(xb).float().squeeze(-1), yb, pos_weight=pw)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            va_auc = roc_auc_score(yva, predict(model, Xva, bs, mean, std))
            if phase == 2:
                if va_auc > best["auc"]:
                    best = {"auc": va_auc,
                            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 5:
                        break
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    res = {"config": cfg_name, "timm_id": timm_id, "size": size, "seed": seed,
           "val_auc": best["auc"],
           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    preds = {}
    for key, X, y in (("cbis_test", Xall[te_mask], yall[te_mask]),
                      ("mias", BANK[size]["mias"], mi["label"].values),
                      ("inbreast", BANK[size]["inbreast"], inb["label"].values)):
        p = predict(model, X, bs, mean, std)
        preds[key] = p
        res[key] = roc_auc_score(y, p)
    del model
    torch.cuda.empty_cache()
    return res, preds


rows, pred_rows = [], []
todo = [c for c in CONFIGS if args.only is None or args.only.lower() in c[0].lower()]
log(f"running {len(todo)} configurations x {len(SEEDS)} seeds on {torch.cuda.get_device_name(0)}")

for cfg_name, timm_id, size, bs in todo:
    for seed in SEEDS:
        ts = time.time()
        try:
            res, preds = run(cfg_name, timm_id, size, bs, seed)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"  {cfg_name} seed {seed}: OOM at bs={bs}, retrying at bs={bs // 2}")
            res, preds = run(cfg_name, timm_id, size, bs // 2, seed)
        res["secs"] = round(time.time() - ts)
        rows.append(res)
        for key, p in preds.items():
            ids = (cb.loc[te_mask, "sample_id"].values if key == "cbis_test"
                   else mi["sample_id"].values if key == "mias" else inb["sample_id"].values)
            grp = (cb.loc[te_mask, "group"].values if key == "cbis_test"
                   else mi["group"].values if key == "mias" else inb["group"].values)
            lab = (cb.loc[te_mask, "label"].values if key == "cbis_test"
                   else mi["label"].values if key == "mias" else inb["label"].values)
            pred_rows.append(pd.DataFrame({"config": cfg_name, "seed": seed, "dataset": key,
                                           "sample_id": ids, "group": grp, "label": lab, "prob": p}))
        log(f"  {cfg_name:22s} seed {seed:5d} ({res['secs']:4d}s) "
            f"val {res['val_auc']:.4f} | test {res['cbis_test']:.4f} | "
            f"MIAS {res['mias']:.4f} | INb {res['inbreast']:.4f}")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "arch_ablation_runs.csv"), index=False)
        pd.concat(pred_rows, ignore_index=True).to_csv(
            os.path.join(OUT, "predictions_arch.csv"), index=False)

R = pd.DataFrame(rows)
agg = R.groupby(["config", "timm_id", "size", "params_M"]).agg(
    cbis_test_mean=("cbis_test", "mean"), cbis_test_std=("cbis_test", "std"),
    mias_mean=("mias", "mean"), inbreast_mean=("inbreast", "mean"),
    val_mean=("val_auc", "mean"), n=("seed", "count")).reset_index()
agg = agg.sort_values("cbis_test_mean", ascending=False)
agg.to_csv(os.path.join(OUT, "arch_ablation_summary.csv"), index=False)
log("=== summary (mean over seeds) ===")
print(agg.to_string(index=False))

with open(os.path.join(OUT, "environment_arch.json"), "w") as f:
    import timm
    json.dump({"torch": torch.__version__, "timm": timm.__version__,
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
               "seeds": SEEDS}, f, indent=2)
log("DONE")

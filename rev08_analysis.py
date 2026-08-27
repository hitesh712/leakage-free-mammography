"""Turn saved per-sample predictions into every statistic the reviewers asked for.

  R1.1 / R5-minor-2  DeLong tests instead of "the CI excludes 0.5"
  R1.2 / R2.2        seed stability over ten runs
  R1.6 / R2.7        confusion matrices computed from predictions
  R1.9 / R2.2        bootstrap CIs at a stated level, patch and cluster
  R2.5               calibration curves, Brier score, ECE
  R5.4               Youden's J recalibration on MIAS, plus label-free transfer

Usage:  python revision/rev08_analysis.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import revstats as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIGS = os.path.join(HERE, "out"), os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

DS_ORDER = ["cbis_test", "mias", "inbreast"]
NICE = {"cbis_val": "CBIS validation", "cbis_test": "CBIS-DDSM (internal test)",
        "mias": "CBIS->MIAS (external)", "inbreast": "CBIS->INbreast (external)"}


def load(tag, fname):
    p = os.path.join(OUT, fname)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d["arm"] = tag
    return d


frames = [f for f in (load("official", "predictions_main.csv"),
                      load("patient_disjoint", "predictions_corrected.csv")) if f is not None]
P = pd.concat(frames, ignore_index=True)
print(f"loaded {len(P)} predictions | arms {sorted(P.arm.unique())} | "
      f"seeds {sorted(P.seed.unique())}")


def ensemble(arm, ds):
    """Seed-averaged probability vector, with labels and cluster ids."""
    d = P[(P.arm == arm) & (P.dataset == ds)]
    g = d.groupby("sample_id", sort=True).agg(prob=("prob", "mean"), label=("label", "first"),
                                              group=("group", "first"))
    return g["label"].values, g["prob"].values, g["group"].values


def per_seed_aucs(arm, ds):
    d = P[(P.arm == arm) & (P.dataset == ds)]
    return np.array([roc_auc_score(s["label"], s["prob"]) for _, s in d.groupby("seed")])


# --------------------------------------------------------------------------
# 1. headline table
# --------------------------------------------------------------------------
rows = []
for arm in P.arm.unique():
    for ds in DS_ORDER:
        if not len(P[(P.arm == arm) & (P.dataset == ds)]):
            continue
        y, p, grp = ensemble(arm, ds)
        aucs = per_seed_aucs(arm, ds)
        auc, lo_d, hi_d = R.delong_ci(y, p)
        lo_b, hi_b = R.bootstrap_ci(y, p, roc_auc_score, n=2000, seed=1)
        lo_c, hi_c = R.bootstrap_ci(y, p, roc_auc_score, groups=grp, n=2000, seed=1)
        m = R.op_metrics(y, p, 0.5)
        rows.append({
            "arm": arm, "dataset": ds, "n": len(y), "n_clusters": len(np.unique(grp)),
            "auc_seed_mean": aucs.mean(), "auc_seed_std": aucs.std(ddof=0),
            "auc_seed_min": aucs.min(), "auc_seed_max": aucs.max(), "n_seeds": len(aucs),
            "auc_ens": auc,
            "delong_lo": lo_d, "delong_hi": hi_d,
            "boot_patch_lo": lo_b, "boot_patch_hi": hi_b,
            "boot_cluster_lo": lo_c, "boot_cluster_hi": hi_c,
            "sens@0.5": m["sens"], "spec@0.5": m["spec"], "acc@0.5": m["acc"],
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            "brier": R.brier(y, p), "ece": R.ece(y, p),
        })
H = pd.DataFrame(rows)
H.to_csv(os.path.join(OUT, "results_headline.csv"), index=False)
print("\n=== headline ===")
print(H[["arm", "dataset", "n", "auc_seed_mean", "auc_seed_std", "auc_ens",
         "boot_cluster_lo", "boot_cluster_hi", "sens@0.5", "spec@0.5", "brier", "ece"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# --------------------------------------------------------------------------
# 2. DeLong tests
# --------------------------------------------------------------------------
tests = []
for arm in P.arm.unique():
    got = [d for d in DS_ORDER if len(P[(P.arm == arm) & (P.dataset == d)])]
    for i in range(len(got)):
        for j in range(i + 1, len(got)):
            y1, p1, _ = ensemble(arm, got[i])
            y2, p2, _ = ensemble(arm, got[j])
            a1, a2, z, pv = R.delong_unpaired(y1, p1, y2, p2)
            tests.append({"arm": arm, "comparison": f"{got[i]} vs {got[j]}", "type": "unpaired",
                          "auc_a": a1, "auc_b": a2, "diff": a1 - a2, "z": z, "p": pv})

# official vs patient-disjoint training, evaluated on the same test sets -> paired
if {"official", "patient_disjoint"} <= set(P.arm.unique()):
    for ds in DS_ORDER:
        y1, p1, _ = ensemble("official", ds)
        y2, p2, _ = ensemble("patient_disjoint", ds)
        if len(y1) != len(y2):
            continue
        a1, a2, z, pv = R.delong_paired(y1, p1, p2)
        tests.append({"arm": "official vs patient_disjoint", "comparison": ds, "type": "paired",
                      "auc_a": a1, "auc_b": a2, "diff": a1 - a2, "z": z, "p": pv})

# architecture comparisons on the shared CBIS test set -> paired
arch_p = os.path.join(OUT, "predictions_arch.csv")
if os.path.exists(arch_p):
    A = pd.read_csv(arch_p)
    At = A[A.dataset == "cbis_test"]
    ref = "EfficientNet-B0 @224"
    if ref in set(At.config):
        base = At[At.config == ref].groupby("sample_id").agg(
            prob=("prob", "mean"), label=("label", "first"))
        for cfg, d in At.groupby("config"):
            if cfg == ref:
                continue
            other = d.groupby("sample_id").agg(prob=("prob", "mean"), label=("label", "first"))
            common = base.index.intersection(other.index)
            a1, a2, z, pv = R.delong_paired(base.loc[common, "label"].values,
                                            base.loc[common, "prob"].values,
                                            other.loc[common, "prob"].values)
            tests.append({"arm": "architecture", "comparison": f"{ref} vs {cfg}", "type": "paired",
                          "auc_a": a1, "auc_b": a2, "diff": a1 - a2, "z": z, "p": pv})

T = pd.DataFrame(tests)
T.to_csv(os.path.join(OUT, "delong_tests.csv"), index=False)
print("\n=== DeLong ===")
print(T.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# --------------------------------------------------------------------------
# 3. threshold transfer
# --------------------------------------------------------------------------
ARM = "official" if "official" in set(P.arm.unique()) else P.arm.unique()[0]
yv, pv_, _ = ensemble(ARM, "cbis_val")
t_src = R.youden_threshold(yv, pv_)
src_prev = float(yv.mean())
tt = []

# Label-free diagnostic: how far has the score distribution actually drifted?
# A deployer can compute this without any target labels, and it indicates
# whether the operating point needs transferring at all.
from scipy import stats as _st
ks_rows = []
for ds in DS_ORDER:
    if not len(P[(P.arm == ARM) & (P.dataset == ds)]):
        continue
    _, p_ds, _ = ensemble(ARM, ds)
    ks = _st.ks_2samp(pv_, p_ds)
    ks_rows.append({"dataset": ds, "ks_vs_source": float(ks.statistic),
                    "ks_p": float(ks.pvalue), "mean_score": float(p_ds.mean()),
                    "source_mean_score": float(pv_.mean())})
KS = pd.DataFrame(ks_rows)
KS.to_csv(os.path.join(OUT, "score_drift.csv"), index=False)
print("\n=== score-distribution drift vs source validation (label-free) ===")
print(KS.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
for ds in DS_ORDER:
    if not len(P[(P.arm == ARM) & (P.dataset == ds)]):
        continue
    y, p, _ = ensemble(ARM, ds)
    prev = float(y.mean())
    strategies = {
        "fixed 0.5": 0.5,
        "Youden J on source validation": t_src,
        "oracle Youden J on target (upper bound)": R.youden_threshold(y, p),
    }
    for name, thr in strategies.items():
        m = R.op_metrics(y, p, thr)
        m.update({"dataset": ds, "strategy": name, "needs_target_labels":
                  name.startswith("oracle"), "prevalence": prev})
        tt.append(m)
    # label-free: map target scores onto the source score distribution by rank
    mapped, thr = R.quantile_transfer(pv_, p, t_src)
    m = R.op_metrics(y, mapped, thr)
    m.update({"dataset": ds, "strategy": "quantile transfer (no target labels)",
              "needs_target_labels": False, "prevalence": prev})
    tt.append(m)
    # prevalence-matched variant: flagged rate implied by the source operating
    # point at the target's prevalence
    thr2, tpr_s, fpr_s = R.prevalence_transfer(yv, pv_, p, t_src, prev)
    m = R.op_metrics(y, p, thr2)
    m.update({"dataset": ds, "strategy": "prevalence-matched transfer (target prevalence known)",
              "needs_target_labels": False, "prevalence": prev})
    tt.append(m)

TT = pd.DataFrame(tt)[["dataset", "strategy", "thr", "sens", "spec", "acc", "youden",
                       "tp", "fp", "tn", "fn", "prevalence", "needs_target_labels"]]
TT.to_csv(os.path.join(OUT, "threshold_transfer.csv"), index=False)
print(f"\n=== threshold transfer (source Youden J = {t_src:.3f}, "
      f"source prevalence {src_prev:.3f}) ===")
print(TT.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# --------------------------------------------------------------------------
# 4. figures: reliability + confusion matrices
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
for ax, ds in zip(axes, DS_ORDER):
    y, p, _ = ensemble(ARM, ds)
    xs, ys, ns = R.reliability(y, p, bins=8)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", color="#c0392b", lw=2, ms=5, label="model")
    ax.set_title(f"{NICE[ds]}\nBrier {R.brier(y, p):.3f} | ECE {R.ece(y, p):.3f}", fontsize=10)
    ax.set_xlabel("predicted probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("observed malignant fraction")
axes[0].legend(fontsize=8)
fig.suptitle("Calibration: discrimination transfers across domains, calibration does not",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "calibration.png"), dpi=250, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=150)
for ax, ds in zip(axes, DS_ORDER):
    y, p, _ = ensemble(ARM, ds)
    m = R.op_metrics(y, p, 0.5)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                color="white" if v > cm.max() * 0.55 else "black")
    ax.set_xticks([0, 1], ["pred B", "pred M"])
    ax.set_yticks([0, 1], ["true B", "true M"])
    ax.set_title(f"{NICE[ds]}\nsens {m['sens']:.2f} | spec {m['spec']:.2f}", fontsize=9)
fig.suptitle("Confusion matrices computed directly from saved predictions "
             "(seed-averaged, threshold 0.5)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "confusion_real.png"), dpi=250, bbox_inches="tight")
plt.close(fig)

with open(os.path.join(OUT, "analysis_meta.json"), "w") as f:
    json.dump({"source_youden_threshold": t_src, "source_prevalence": src_prev,
               "arms": sorted(P.arm.unique()), "seeds": sorted(int(s) for s in P.seed.unique())},
              f, indent=2)
print("\nwrote results_headline.csv, delong_tests.csv, threshold_transfer.csv, "
      "figures/calibration.png, figures/confusion_real.png")

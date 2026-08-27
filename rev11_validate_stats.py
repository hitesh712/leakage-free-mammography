"""Independent validation of the hand-written DeLong implementation.

Three tables and several significance claims in the paper rest on `revstats`,
which implements the Sun & Xu (2014) fast DeLong algorithm from scratch. Nothing
external had checked it, so this script does:

  1. AUC agreement with scikit-learn on the real prediction files (exact).
  2. DeLong standard error against a 20,000-resample bootstrap SE, on the real
     data. These are independent estimators of the same quantity and should
     agree closely.
  3. Degenerate case: comparing a score vector with itself must give z = 0.
  4. Null calibration by simulation: when two models genuinely have equal AUC,
     a valid test rejects at the nominal 5%. Run for both the paired variant
     (correlated curves, shared cases) and the unpaired variant (independent
     cohorts), which is the one used for internal-versus-external comparisons
     and the less standard of the two.
  5. Power check: a real difference must actually be detected.
  6. Tie handling, since midrank correctness is where such implementations
     usually break.

Usage:  python revision/rev11_validate_stats.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import revstats as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
rng = np.random.default_rng(12345)
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
print("\n1. AUC agreement with scikit-learn (real prediction files)")
P = pd.read_csv(os.path.join(OUT, "predictions_main.csv"))
real = {}
for ds, g in P[P.dataset.isin(["cbis_test", "mias", "inbreast"])].groupby("dataset"):
    a = g.groupby("sample_id").agg(prob=("prob", "mean"), label=("label", "first"))
    y, p = a["label"].values, a["prob"].values
    real[ds] = (y, p)
    auc_dl, var = R.delong_auc_var(y, p)
    auc_sk = roc_auc_score(y, p)
    check(f"{ds:10s} DeLong AUC == sklearn", abs(auc_dl - auc_sk) < 1e-9,
          f"DeLong {auc_dl:.10f}  sklearn {auc_sk:.10f}")

# --------------------------------------------------------------------------
print("\n2. DeLong SE vs bootstrap SE (20,000 resamples, real data)")
for ds, (y, p) in real.items():
    _, var = R.delong_auc_var(y, p)
    se_dl = np.sqrt(var)
    idx = np.arange(len(y))
    boots = []
    for _ in range(20000):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        boots.append(roc_auc_score(y[s], p[s]))
    se_bs = float(np.std(boots, ddof=1))
    rel = abs(se_dl - se_bs) / se_bs
    check(f"{ds:10s} SE within 10% of bootstrap", rel < 0.10,
          f"DeLong {se_dl:.5f}  bootstrap {se_bs:.5f}  rel.diff {100 * rel:.1f}%")

# --------------------------------------------------------------------------
print("\n3. Degenerate case: a score vector compared with itself")
y, p = real["cbis_test"]
a1, a2, z, pv = R.delong_paired(y, p, p)
check("paired self-comparison gives z = 0", abs(z) < 1e-8, f"z={z:.2e}, p={pv:.4f}")
check("paired self-comparison gives equal AUCs", abs(a1 - a2) < 1e-12)

# --------------------------------------------------------------------------
print("\n4. Null calibration by simulation (nominal alpha = 0.05)")


def sim_null_paired(reps=2000, n=500, prev=0.4):
    """Two correlated scorers with identical true AUC."""
    reject = 0
    for _ in range(reps):
        y = (rng.random(n) < prev).astype(int)
        signal = y + rng.normal(0, 1.0, n)          # shared underlying signal
        s1 = signal + rng.normal(0, 0.7, n)
        s2 = signal + rng.normal(0, 0.7, n)         # same noise scale -> equal AUC
        _, _, _, pv = R.delong_paired(y, s1, s2)
        reject += pv < 0.05
    return reject / reps


def sim_null_unpaired(reps=2000, n=400, prev=0.4):
    """Two independent cohorts scored by equally good models."""
    reject = 0
    for _ in range(reps):
        y1 = (rng.random(n) < prev).astype(int)
        y2 = (rng.random(n) < prev).astype(int)
        s1 = y1 + rng.normal(0, 1.0, n)
        s2 = y2 + rng.normal(0, 1.0, n)
        _, _, _, pv = R.delong_unpaired(y1, s1, y2, s2)
        reject += pv < 0.05
    return reject / reps


r_p = sim_null_paired()
# binomial 95% band on 2000 reps at alpha=0.05 is roughly 0.040-0.060
check("paired: type-I error near 5%", 0.035 <= r_p <= 0.065, f"rejection rate {r_p:.3f}")
r_u = sim_null_unpaired()
check("unpaired: type-I error near 5%", 0.035 <= r_u <= 0.065, f"rejection rate {r_u:.3f}")

# --------------------------------------------------------------------------
print("\n5. Power: a genuine difference must be detected")


def sim_power(reps=400, n=500, prev=0.4):
    det = 0
    for _ in range(reps):
        y = (rng.random(n) < prev).astype(int)
        sig = y + rng.normal(0, 1.0, n)
        s_good = sig + rng.normal(0, 0.3, n)
        s_bad = sig + rng.normal(0, 2.0, n)          # clearly worse
        _, _, _, pv = R.delong_paired(y, s_good, s_bad)
        det += pv < 0.05
    return det / reps


pw = sim_power()
check("paired: detects a large true difference", pw > 0.90, f"power {pw:.3f}")

# --------------------------------------------------------------------------
print("\n6. Tie handling (midrank correctness)")
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
p_tied = np.array([0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8])
auc_dl, _ = R.delong_auc_var(y, p_tied)
check("heavily tied scores: AUC == sklearn", abs(auc_dl - roc_auc_score(y, p_tied)) < 1e-9,
      f"DeLong {auc_dl:.6f}  sklearn {roc_auc_score(y, p_tied):.6f}")

p_const = np.full(8, 0.5)
auc_c, _ = R.delong_auc_var(y, p_const)
check("all-constant scores give AUC 0.5", abs(auc_c - 0.5) < 1e-9, f"AUC {auc_c:.6f}")

# --------------------------------------------------------------------------
print("\n7. Reported p-values reproduced from the released predictions")
y_i, p_i = real["cbis_test"]
y_m, p_m = real["mias"]
_, _, z, pv = R.delong_unpaired(y_i, p_i, y_m, p_m)
check("internal vs MIAS reproduces the reported p", abs(pv - 0.818) < 0.005,
      f"p = {pv:.4f} (paper reports 0.82)")

print()
if FAILURES:
    print(f"VALIDATION FAILED: {len(FAILURES)} check(s) -> {FAILURES}")
    raise SystemExit(1)
print("ALL CHECKS PASSED")

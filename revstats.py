"""Statistical primitives for the revision: DeLong, cluster bootstrap, calibration."""
import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, float)
    out[J] = T
    return out


def _fast_delong(preds_sorted, m):
    """Sun & Xu (2014) fast DeLong. `preds_sorted` is (k, n) with the m positives first."""
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def _prep(y, *scores):
    y = np.asarray(y)
    order = np.argsort(-y, kind="mergesort")   # positives first, order preserved
    m = int(y.sum())
    return np.vstack([np.asarray(s)[order] for s in scores]), m


def delong_auc_var(y, score):
    """AUC and its DeLong variance for a single ROC curve."""
    P, m = _prep(y, score)
    auc, cov = _fast_delong(P, m)
    return float(auc[0]), float(cov[0, 0])


def delong_paired(y, score_a, score_b):
    """Two ROC curves on the SAME samples. Returns (auc_a, auc_b, z, p)."""
    P, m = _prep(y, score_a, score_b)
    aucs, cov = _fast_delong(P, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))


def delong_unpaired(y1, s1, y2, s2):
    """Two ROC curves on DIFFERENT samples (e.g. internal vs external test set).

    DeLong's test proper is for correlated curves; for independent samples the
    standard analogue combines the two DeLong variances additively.
    """
    a1, v1 = delong_auc_var(y1, s1)
    a2, v2 = delong_auc_var(y2, s2)
    z = (a1 - a2) / np.sqrt(v1 + v2)
    return a1, a2, float(z), float(2 * stats.norm.sf(abs(z)))


def delong_ci(y, score, alpha=0.05):
    auc, var = delong_auc_var(y, score)
    half = stats.norm.ppf(1 - alpha / 2) * np.sqrt(var)
    return auc, max(0.0, auc - half), min(1.0, auc + half)


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def bootstrap_ci(y, score, fn, groups=None, n=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI.

    With `groups`, resampling is at the cluster level (patients or source
    images) rather than the individual patch level, so the interval reflects the
    fact that patches from one patient are not independent.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    score = np.asarray(score)
    vals = []
    if groups is None:
        idx = np.arange(len(y))
        for _ in range(n):
            s = rng.choice(idx, len(idx), replace=True)
            if len(np.unique(y[s])) < 2:
                continue
            vals.append(fn(y[s], score[s]))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n):
            gs = rng.choice(uniq, len(uniq), replace=True)
            s = np.concatenate([members[g] for g in gs])
            if len(np.unique(y[s])) < 2:
                continue
            vals.append(fn(y[s], score[s]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, bins=10):
    """Expected calibration error, equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e, N = 0.0, len(y)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / N * abs(y[m].mean() - p[m].mean())
    return float(e)


def reliability(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def youden_threshold(y, p):
    """Threshold maximising sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(thr[j])


def op_metrics(y, p, thr):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {"thr": float(thr), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sens": sens, "spec": spec, "acc": (tp + tn) / max(len(y), 1),
            "youden": sens + spec - 1}


def quantile_transfer(src_scores, tgt_scores, src_thr):
    """Unsupervised cross-domain threshold transfer.

    Discrimination transfers across sites but calibration does not, so a
    threshold tuned on the source silently drifts on a new scanner. This maps
    the target scores onto the source score distribution by rank -- using no
    target labels, only the unlabelled target scores a deployed system already
    has -- and then applies the source-calibrated threshold.

    The mapping is monotone, so AUC is untouched; only the operating point moves.
    Assumes source and target class prevalence are comparable; where they are
    not, `prevalence_transfer` is the appropriate variant.
    """
    src = np.sort(np.asarray(src_scores))
    tgt = np.asarray(tgt_scores)
    ranks = np.searchsorted(np.sort(tgt), tgt, side="right") / len(tgt)
    mapped = np.quantile(src, np.clip(ranks, 0, 1))
    return mapped, float(src_thr)


def prevalence_transfer(src_labels, src_scores, tgt_scores, src_thr, tgt_prev):
    """Quantile transfer with a prevalence-matched flagged rate.

    Plain rank matching forces the target to flag the same *fraction* of cases
    as the source. That is only right when the two cohorts have the same class
    prevalence, because the source flagged rate is itself
    $\\mathrm{TPR}\\pi + \\mathrm{FPR}(1-\\pi)$ and so moves with $\\pi$.

    Sensitivity and specificity, unlike the flagged rate, do not depend on
    prevalence. So we read TPR and FPR off the labelled *source* validation fold
    at its own Youden threshold, and ask what fraction that same operating point
    would flag in a cohort whose prevalence is the target's:

        flagged_tgt = TPR_src * pi_tgt + FPR_src * (1 - pi_tgt)

    then take the corresponding upper quantile of the target scores. Only the
    target prevalence is needed -- an epidemiological quantity, not case-level
    labels.
    """
    y = np.asarray(src_labels)
    s = np.asarray(src_scores)
    pred = s >= src_thr
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else 0.0
    fpr = float(pred[y == 0].mean()) if (y == 0).any() else 0.0
    flagged_tgt = float(np.clip(tpr * tgt_prev + fpr * (1.0 - tgt_prev), 0.01, 0.99))
    return float(np.quantile(np.asarray(tgt_scores), 1.0 - flagged_tgt)), tpr, fpr

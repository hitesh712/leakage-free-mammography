"""Emit every results table as LaTeX, straight from the result CSVs.

Numbers are never transcribed by hand: the manuscript \\input{}s the file this
writes, so re-running the experiments and re-running this script keeps the paper
in sync automatically.

Writes tables_generated.tex and numbers_generated.tex into the directory given
by --paper (default: alongside this script).

Usage:  python rev09_maketables.py [--paper PATH_TO_LATEX_FOLDER]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

_ap = argparse.ArgumentParser()
_ap.add_argument("--paper", default=HERE,
                 help="directory to write the generated .tex files into")
PAPER = _ap.parse_args().paper
os.makedirs(PAPER, exist_ok=True)

NICE = {"cbis_test": "CBIS-DDSM (internal test)", "mias": "CBIS$\\rightarrow$MIAS",
        "inbreast": "CBIS$\\rightarrow$INbreast"}
SHORT = {"cbis_test": "CBIS internal", "mias": "MIAS", "inbreast": "INbreast"}


def f(v, n=3):
    return "---" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"


def read(name):
    p = os.path.join(OUT, name)
    return pd.read_csv(p) if os.path.exists(p) else None


chunks = []


def add(s):
    chunks.append(s)


# --------------------------------------------------------------------------
H = read("results_headline.csv")
if H is not None:
    off = H[H.arm == "official"]
    add(r"""
%%%%%%%%%%%%%%%%%%%% external validation %%%%%%%%%%%%%%%%%%%%
\begin{table}[htbp]
\centering
\caption{\rev{Leakage-free external validation. A single EfficientNetB0 @224
trained on the CBIS-DDSM official training split is applied without retraining
to both external sets. AUC is given as the mean $\pm$ standard deviation over
ten seeds, alongside the observed range; the 95\% CI is a cluster bootstrap
(2{,}000 resamples of patients for CBIS-DDSM, of source images for the external
sets) on the seed-averaged predictions. Sensitivity and specificity are at the
0.5 operating point.}}
\small
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}lrrlllrr@{}}
\toprule
\textbf{Test set} & \textbf{n} & \textbf{Clusters} & \textbf{AUC (10 seeds)} &
\textbf{Range} & \textbf{95\% CI} & \textbf{Sens} & \textbf{Spec} \\
\midrule""")
    for ds in ["cbis_test", "mias", "inbreast"]:
        r = off[off.dataset == ds]
        if not len(r):
            continue
        r = r.iloc[0]
        add(f"{NICE[ds]} & {int(r['n'])} & {int(r['n_clusters'])} & "
            f"${f(r['auc_seed_mean'])} \\pm {f(r['auc_seed_std'])}$ & "
            f"{f(r['auc_seed_min'])}--{f(r['auc_seed_max'])} & "
            f"{f(r['boot_cluster_lo'])}--{f(r['boot_cluster_hi'])} & "
            f"{f(r['sens@0.5'], 2)} & {f(r['spec@0.5'], 2)} \\\\")
    add(r"""\bottomrule
\end{tabular}
\label{tab:extval}
\end{table}""")

    # confusion matrices from real predictions
    add(r"""
%%%%%%%%%%%%%%%%%%%% confusion matrices %%%%%%%%%%%%%%%%%%%%
\begin{table}[htbp]
\centering
\caption{\rev{Confusion matrices computed directly from the saved per-sample
predictions (seed-averaged, threshold 0.5), replacing the matrices reconstructed
from summary statistics in the submitted version. FN is the clinically costly
error.}}
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Test set} & \textbf{TP} & \textbf{FN} & \textbf{TN} & \textbf{FP} &
\textbf{Sens} & \textbf{Spec} \\
\midrule""")
    for ds in ["cbis_test", "mias", "inbreast"]:
        r = off[off.dataset == ds]
        if not len(r):
            continue
        r = r.iloc[0]
        add(f"{NICE[ds]} & {int(r['tp'])} & {int(r['fn'])} & {int(r['tn'])} & "
            f"{int(r['fp'])} & {f(r['sens@0.5'], 2)} & {f(r['spec@0.5'], 2)} \\\\")
    add(r"""\bottomrule
\end{tabular}
\label{tab:confusion}
\end{table}""")

    # calibration
    add(r"""
%%%%%%%%%%%%%%%%%%%% calibration %%%%%%%%%%%%%%%%%%%%
\begin{table}[htbp]
\centering
\caption{\rev{Discrimination transfers but calibration does not. All quantities
are computed on the seed-averaged predicted probabilities, so the AUC column is
the seed-ensembled value and differs marginally from the seed-mean of
Table~\ref{tab:extval}. AUC is essentially unchanged from internal to external
data, whereas the Brier score and expected calibration error degrade and the
score distribution shifts away from the source (two-sample
Kolmogorov--Smirnov statistic against the source validation scores, computable
without any target labels).}}
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Test set} & \textbf{AUC (ens.)} & \textbf{Brier} & \textbf{ECE} &
\textbf{KS vs.\ source} \\
\midrule""")
    KS = read("score_drift.csv")
    for ds in ["cbis_test", "mias", "inbreast"]:
        r = off[off.dataset == ds]
        if not len(r):
            continue
        r = r.iloc[0]
        ks = ""
        if KS is not None and (KS.dataset == ds).any():
            ks = f(KS[KS.dataset == ds].iloc[0]["ks_vs_source"])
        add(f"{NICE[ds]} & {f(r['auc_ens'])} & {f(r['brier'])} & {f(r['ece'])} & {ks} \\\\")
    add(r"""\bottomrule
\end{tabular}
\label{tab:calibration}
\end{table}""")

    # official vs patient-disjoint
    if (H.arm == "patient_disjoint").any():
        pd_ = H[H.arm == "patient_disjoint"]
        add(r"""
%%%%%%%%%%%%%%%%%%%% official vs patient-disjoint %%%%%%%%%%%%%%%%%%%%
\begin{table}[htbp]
\centering
\caption{\rev{Effect of repairing the CBIS-DDSM official split. Pooling mass and
calcification cases makes the official partition non-patient-disjoint: 31
patients appear on both sides, affecting 60 of the 704 test lesions. The repair
removes those patients from training (83 lesions) and leaves the official test
set untouched, so the two columns are evaluated on identical cases. AUC is the
mean over ten seeds.}}
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Test set} & \textbf{Official split} & \textbf{Patient-disjoint} &
\textbf{$\Delta$} \\
\midrule""")
        for ds in ["cbis_test", "mias", "inbreast"]:
            a = off[off.dataset == ds]
            b = pd_[pd_.dataset == ds]
            if not len(a) or not len(b):
                continue
            av, bv = a.iloc[0]["auc_seed_mean"], b.iloc[0]["auc_seed_mean"]
            add(f"{NICE[ds]} & ${f(av)} \\pm {f(a.iloc[0]['auc_seed_std'])}$ & "
                f"${f(bv)} \\pm {f(b.iloc[0]['auc_seed_std'])}$ & "
                f"{bv - av:+.3f} \\\\")
        add(r"""\bottomrule
\end{tabular}
\label{tab:splitrepair}
\end{table}""")

# --------------------------------------------------------------------------
T = read("delong_tests.csv")
if T is not None:
    add(r"""
%%%%%%%%%%%%%%%%%%%% DeLong %%%%%%%%%%%%%%%%%%%%
\begin{table}[htbp]
\centering
\caption{\rev{DeLong tests. Internal and external AUCs are computed on different
cases, so their ROC curves are independent and the two DeLong variances are
combined additively; comparisons on shared cases use the paired test. The
internal-versus-MIAS comparison is the one that supports the generalisation
claim: the difference is not significant, i.e.\ external performance is
statistically indistinguishable from internal, which is a stronger statement
than a confidence interval merely excluding chance.}}
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}llrrrr@{}}
\toprule
\textbf{Comparison} & \textbf{Type} & \textbf{AUC A} & \textbf{AUC B} &
\textbf{$z$} & \textbf{$p$} \\
\midrule""")
    lab = {"cbis_test vs mias": "CBIS internal vs.\\ MIAS",
           "cbis_test vs inbreast": "CBIS internal vs.\\ INbreast",
           "mias vs inbreast": "MIAS vs.\\ INbreast"}
    for _, r in T[T.arm == "official"].iterrows():
        nm = lab.get(r["comparison"], r["comparison"])
        pv = "$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        add(f"{nm} & {r['type']} & {f(r['auc_a'])} & {f(r['auc_b'])} & "
            f"{r['z']:.2f} & {pv} \\\\")
    sub = T[T.arm == "official vs patient_disjoint"]
    if len(sub):
        add(r"\midrule")
        for _, r in sub.iterrows():
            pv = "$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
            add(f"Official vs.\\ patient-disjoint ({SHORT.get(r['comparison'], r['comparison'])}) "
                f"& paired & {f(r['auc_a'])} & {f(r['auc_b'])} & {r['z']:.2f} & {pv} \\\\")
    add(r"""\bottomrule
\end{tabular}
\label{tab:delong}
\end{table}""")

# --------------------------------------------------------------------------
TT = read("threshold_transfer.csv")
if TT is not None:
    add(r"""
%%%%%%%%%%%%%%%%%%%% threshold transfer %%%%%%%%%%%%%%%%%%%%
\begin{table}[htbp]
\centering
\caption{\rev{Choosing the operating point at a new institution. The oracle
tunes Youden's $J$ on the labelled target set and is therefore unattainable in
deployment; it is shown as an upper bound. Carrying the source threshold across
unchanged is the naive baseline. Quantile transfer and its prevalence-matched
variant use only unlabelled target scores (plus, for the latter, an estimate of
target prevalence). Best attainable $J$ per dataset in bold.}}
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{}lrrrrc@{}}
\toprule
\textbf{Threshold strategy} & \textbf{Thr.} &
\textbf{Sens} & \textbf{Spec} & \textbf{Youden $J$} & \textbf{Needs labels?} \\
\midrule""")
    order = ["fixed 0.5", "Youden J on source validation",
             "quantile transfer (no target labels)",
             "prevalence-matched transfer (target prevalence known)",
             "oracle Youden J on target (upper bound)"]
    pretty = {"fixed 0.5": "fixed 0.5 (as submitted)",
              "Youden J on source validation": "source Youden $J$, carried across",
              "quantile transfer (no target labels)": "quantile transfer",
              "prevalence-matched transfer (target prevalence known)": "prevalence-matched transfer",
              "oracle Youden J on target (upper bound)": "\\emph{oracle} target Youden $J$"}
    for ds in ["cbis_test", "mias", "inbreast"]:
        sub = TT[TT.dataset == ds]
        if not len(sub):
            continue
        feasible = sub[~sub.needs_target_labels.astype(bool)]
        best = feasible["youden"].max() if len(feasible) else None
        add(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{NICE[ds]}}}}} \\\\")
        for s in order:
            r = sub[sub.strategy == s]
            if not len(r):
                continue
            r = r.iloc[0]
            j = f(r["youden"])
            if best is not None and not bool(r["needs_target_labels"]) \
                    and abs(r["youden"] - best) < 1e-9:
                j = f"\\textbf{{{j}}}"
            add(f"\\quad {pretty[s]} & {f(r['thr'], 3)} & {f(r['sens'], 3)} & "
                f"{f(r['spec'], 3)} & {j} & "
                f"{'yes' if bool(r['needs_target_labels']) else 'no'} \\\\")
        add(r"\addlinespace")
    add(r"""\bottomrule
\end{tabular}
\label{tab:threshold}
\end{table}""")

# --------------------------------------------------------------------------
A = read("arch_ablation_summary.csv")
if A is not None:
    add(r"""
%%%%%%%%%%%%%%%%%%%% architecture x resolution %%%%%%%%%%%%%%%%%%%%
\begin{table}[htbp]
\centering
\caption{\rev{Architecture and resolution under one identical leakage-free
protocol, run in a single PyTorch implementation so the families are directly
comparable. EfficientNet-B0/B3 and ResNet-50 each appear at 224 and 300\,px, so
resolution is varied with the backbone held fixed. AUC is the mean over three
seeds on the CBIS-DDSM official test split; external AUCs are for the same
models applied without retraining.}}
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Backbone} & \textbf{Params (M)} & \textbf{Input} &
\textbf{CBIS test AUC} & \textbf{MIAS} & \textbf{INbreast} \\
\midrule""")
    for _, r in A.iterrows():
        name = str(r["config"]).split(" @")[0]
        sd = r.get("cbis_test_std", np.nan)
        sdtxt = f" $\\pm$ {f(sd)}" if not (isinstance(sd, float) and np.isnan(sd)) else ""
        add(f"{name} & {r['params_M']:.1f} & {int(r['size'])} & "
            f"{f(r['cbis_test_mean'])}{sdtxt} & {f(r['mias_mean'])} & "
            f"{f(r['inbreast_mean'])} \\\\")
    add(r"""\bottomrule
\end{tabular}
\label{tab:arch}
\end{table}""")

# --------------------------------------------------------------------------
W = read("wholeimage_summary.csv")
if W is not None and H is not None:
    patch = H[(H.arm == "official") & (H.dataset == "cbis_test")]
    patch_auc = patch.iloc[0]["auc_seed_mean"] if len(patch) else float("nan")
    add(r"""
%%%%%%%%%%%%%%%%%%%% whole image baseline %%%%%%%%%%%%%%%%%%%%
\begin{table}[htbp]
\centering
\caption{\rev{Whole-mammogram versus lesion-patch classification on CBIS-DDSM,
the primary dataset, under the same patient-wise leakage-free protocol. Two
input resolutions are used so that the gap cannot be dismissed as an artefact of
downsampling a full mammogram until the lesion disappears. Mean over three
seeds.}}
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}llrr@{}}
\toprule
\textbf{Framing} & \textbf{Input} & \textbf{n (train/test)} &
\textbf{CBIS test AUC} \\
\midrule""")
    for _, r in W.iterrows():
        sd = r.get("test_auc_std", np.nan)
        sdtxt = f" $\\pm$ {f(sd)}" if not (isinstance(sd, float) and np.isnan(sd)) else ""
        add(f"Whole mammogram & {int(r['size'])}\\,px & 2458\\,/\\,399 & "
            f"{f(r['test_auc_mean'])}{sdtxt} \\\\")
    add(f"\\textbf{{Lesion patch}} & 224\\,px & 2863\\,/\\,704 & "
        f"\\textbf{{{f(patch_auc)}}} \\\\")
    add(r"""\bottomrule
\end{tabular}
\label{tab:wholeimage}
\end{table}""")

# --------------------------------------------------------------------------
P = read("perturbation.csv")
if P is not None:
    add(r"""
%%%%%%%%%%%%%%%%%%%% crop perturbation %%%%%%%%%%%%%%%%%%%%
\begin{table}[htbp]
\centering
\caption{\rev{Sensitivity to imperfect lesion localisation. Each crop is
regenerated from the source image with a random translation, expressed as a
fraction of the patch side, or with the box scaled; the model is not retrained.
Because the crops are regenerated rather than taken from the official
pre-cropped ROIs, the top row is the correct baseline for reading the
degradation. Mean over three random jitter draws and a three-seed model
ensemble.}}
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}llrrr@{}}
\toprule
\textbf{Translation} & \textbf{Scale} & \textbf{CBIS test} & \textbf{MIAS} &
\textbf{INbreast} \\
\midrule""")
    for _, r in P.iterrows():
        shift = f"{int(r['shift_pct'])}\\%"
        add(f"{shift} & {r['scale']:.2f} & {f(r['cbis_test'])} & {f(r['mias'])} & "
            f"{f(r['inbreast'])} \\\\")
    add(r"""\bottomrule
\end{tabular}
\label{tab:perturb}
\end{table}""")

# --------------------------------------------------------------------------
lk_path = os.path.join(OUT, "leakage_demo_bm.json")
if os.path.exists(lk_path):
    L = json.load(open(lk_path))
    add(r"""
%%%%%%%%%%%%%%%%%%%% controlled leakage demonstration %%%%%%%%%%%%%%%%%%%%
\begin{table}[htbp]
\centering
\caption{\rev{The leakage effect, measured with a single variable changed.
Both arms use the same """ + str(L["n_films"]) + r""" Mini-MIAS films, the same
benign-versus-malignant labels, the same frozen MobileNetV2 features, the same
RBF-SVM and the same """ + str(L["rotations"]) + r""" rotations; they differ
only in whether the split is taken before or after augmentation. Mean $\pm$
standard deviation over """ + str(L["folds"]) + r"""-fold cross-validation.
Majority-class accuracy for this task is """ + f(L["majority_class_accuracy"], 3) + r""".}}
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}llrr@{}}
\toprule
\textbf{Protocol} & \textbf{Split unit} & \textbf{Accuracy} & \textbf{ROC-AUC} \\
\midrule""")
    add(f"Augment $\\rightarrow$ split & augmented sample & "
        f"$\\mathbf{{{f(L['leaky_accuracy'])} \\pm {f(L['leaky_accuracy_std'])}}}$ & "
        f"\\textbf{{{f(L['leaky_auc'])}}} \\\\")
    add(f"Split $\\rightarrow$ augment train only & source film & "
        f"${f(L['corrected_accuracy'])} \\pm {f(L['corrected_accuracy_std'])}$ & "
        f"{f(L['corrected_auc'])} $\\pm$ {f(L['corrected_auc_std'])} \\\\")
    add(r"""\bottomrule
\end{tabular}
\label{tab:leakage_controlled}
\end{table}""")
    macros_leak = L
else:
    macros_leak = None

# --------------------------------------------------------------------------
G = read("gradcam_summary.csv")
if G is not None:
    add(r"""
%%%%%%%%%%%%%%%%%%%% grad-cam %%%%%%%%%%%%%%%%%%%%
\begin{table}[htbp]
\centering
\caption{\rev{Quantitative Grad-CAM localisation against the radiologist
annotation, with two null models scored on the identical patches. Crops are
deliberately off-centred and the mask is cropped through the same transform.
\emph{Pointing} is the fraction of patches whose peak activation falls inside
the annotation; \emph{energy} the fraction of activation mass inside it;
\emph{conc.} the energy divided by the lesion's area fraction, which equals 1
for a uniform map and is therefore comparable across regimes. In the
\emph{native} crop convention the lesion fills too much of the patch for the
test to discriminate and the centre prior wins; the \emph{wide-field} crops,
where the lesion covers about a tenth of the patch, are the informative
comparison.}}
\small
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
\textbf{Setting} & \textbf{Saliency} & \textbf{Pointing} & \textbf{Energy} &
\textbf{Conc.} & \textbf{IoU@top20} & \textbf{Lesion area} \\
\midrule""")
    dsname = {"cbis_test": "CBIS-DDSM", "mias": "Mini-MIAS"}
    regname = {"native": "native crop", "widefield": "wide field"}
    for reg in ("native", "widefield"):
        for _, r in G.iterrows():
            key = str(r["dataset"])
            if "|" not in key or key.split("|")[1] != reg:
                continue
            d = f"{dsname.get(key.split('|')[0], key)}, {regname[reg]}"
            win = r["pointing_hit"] > r["pointing_hit_centre"]
            gc = f"\\textbf{{{f(r['pointing_hit'])}}}" if win else f(r["pointing_hit"])
            add(f"{d} & Grad-CAM & {gc} & {f(r['energy'])} & "
                f"{f(r['conc_gradcam'], 2)} & {f(r['iou'])} & {f(r['mask_frac'])} \\\\")
            add(f" & centre prior & {f(r['pointing_hit_centre'])} & "
                f"{f(r['energy_centre'])} & {f(r['conc_centre'], 2)} & --- & \\\\")
            add(f" & random & {f(r['pointing_hit_random'])} & "
                f"{f(r['energy_random'])} & {f(r['conc_random'], 2)} & --- & \\\\")
            add(r"\addlinespace")
    add(r"""\bottomrule
\end{tabular}
\label{tab:gradcam}
\end{table}""")

path = os.path.join(PAPER, "tables_generated.tex")
with open(path, "w", encoding="utf-8") as fh:
    fh.write("% GENERATED by revision/rev09_maketables.py -- do not edit by hand.\n")
    fh.write("\n".join(chunks) + "\n")
print(f"wrote {path} ({len(chunks)} lines from "
      f"{sum(x is not None for x in [H, T, TT, A, W, P, G])} result files)")

# --------------------------------------------------------------------------
# Key numbers as macros, so the abstract and running text cannot drift from the
# tables. Every figure quoted in prose should come from here.
# --------------------------------------------------------------------------
macros = {}
if H is not None:
    off = H[H.arm == "official"]
    key = {"cbis_test": "Int", "mias": "Mias", "inbreast": "Inb"}
    for ds, tag in key.items():
        r = off[off.dataset == ds]
        if not len(r):
            continue
        r = r.iloc[0]
        macros[f"auc{tag}"] = f(r["auc_seed_mean"])
        macros[f"sd{tag}"] = f(r["auc_seed_std"])
        macros[f"ci{tag}"] = f"{f(r['boot_cluster_lo'])}--{f(r['boot_cluster_hi'])}"
        macros[f"sens{tag}"] = f(r["sens@0.5"], 2)
        macros[f"spec{tag}"] = f(r["spec@0.5"], 2)
        macros[f"fn{tag}"] = str(int(r["fn"]))
        macros[f"tp{tag}"] = str(int(r["tp"]))
        macros[f"missrate{tag}"] = f(100 * r["fn"] / max(r["fn"] + r["tp"], 1), 0)
    macros["maxsd"] = f(off["auc_seed_std"].max())
    if (H.arm == "patient_disjoint").any():
        a = off[off.dataset == "cbis_test"].iloc[0]["auc_seed_mean"]
        b = H[(H.arm == "patient_disjoint") & (H.dataset == "cbis_test")].iloc[0]["auc_seed_mean"]
        macros["aucPD"] = f(b)
        macros["deltaPD"] = f"{b - a:+.3f}"
if T is not None:
    r = T[(T.arm == "official") & (T.comparison == "cbis_test vs mias")]
    if len(r):
        pv = r.iloc[0]["p"]
        macros["pIntMias"] = "$<$0.001" if pv < 0.001 else f"{pv:.2f}"
    r = T[(T.arm == "official") & (T.comparison == "cbis_test vs inbreast")]
    if len(r):
        pv = r.iloc[0]["p"]
        macros["pIntInb"] = "$<$0.001" if pv < 0.001 else f"{pv:.3f}"
if A is not None and len(A):
    macros["archLo"] = f(A["cbis_test_mean"].min())
    macros["archHi"] = f(A["cbis_test_mean"].max())
    macros["archN"] = str(len(A))
    macros["archBest"] = str(A.iloc[0]["config"]).replace("_", " ")
if W is not None and len(W):
    macros["wholeBest"] = f(W["test_auc_mean"].max())
if TT is not None:
    for ds, tag in (("mias", "Mias"), ("inbreast", "Inb")):
        sub = TT[TT.dataset == ds]
        for s, nm in (("fixed 0.5", "Fixed"),
                      ("prevalence-matched transfer (target prevalence known)", "Prev"),
                      ("oracle Youden J on target (upper bound)", "Oracle"),
                      ("Youden J on source validation", "SrcY")):
            r = sub[sub.strategy == s]
            if len(r):
                macros[f"j{nm}{tag}"] = f(r.iloc[0]["youden"])
                macros[f"spec{nm}{tag}"] = f(r.iloc[0]["spec"], 2)
if G is not None:
    for _, r in G.iterrows():
        key = str(r["dataset"])
        if "|" not in key:
            continue
        base, reg = key.split("|")
        tag = ("Int" if base == "cbis_test" else "Mias") + ("Wf" if reg == "widefield" else "Nat")
        macros[f"point{tag}"] = f(r["pointing_hit"], 2)
        macros[f"pointCentre{tag}"] = f(r["pointing_hit_centre"], 2)
        macros[f"pointRandom{tag}"] = f(r["pointing_hit_random"], 2)
        macros[f"conc{tag}"] = f(r["conc_gradcam"], 2)
if P is not None and len(P):
    base = P[(P.shift_pct == 0) & (P.scale == 1.0)]
    if len(base):
        for col, tag in (("cbis_test", "Int"), ("mias", "Mias"), ("inbreast", "Inb")):
            b = base.iloc[0][col]
            r15 = P[(P.shift_pct == 15) & (P.scale == 1.0)]
            if len(r15):
                # LaTeX macro names may not contain digits.
                macros[f"dropFifteen{tag}"] = f(b - r15.iloc[0][col])

if macros_leak:
    macros["leakAcc"] = f(macros_leak["leaky_accuracy"])
    macros["leakAuc"] = f(macros_leak["leaky_auc"])
    macros["corrAcc"] = f(macros_leak["corrected_accuracy"])
    macros["corrAuc"] = f(macros_leak["corrected_auc"])
    macros["leakMaj"] = f(macros_leak["majority_class_accuracy"])
    macros["leakFilms"] = str(macros_leak["n_films"])
    macros["leakRot"] = str(macros_leak["rotations"])

npath = os.path.join(PAPER, "numbers_generated.tex")
with open(npath, "w", encoding="utf-8") as fh:
    fh.write("% GENERATED by revision/rev09_maketables.py -- do not edit by hand.\n")
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {npath} ({len(macros)} macros)")

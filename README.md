# Leakage-free patch-based mammography classification

Code, data splits and per-sample predictions for:

> H. Kag, D. Kumar and A. Diwan, "Breast Lesion Classification: Leakage-Free
> Patch-Based Transfer Learning with Cross-Dataset Evaluation on CBIS-DDSM,
> MIAS and INbreast," *Jordanian Journal of Computers and Information
> Technology (JJCIT)*.

Each number in the paper comes from one of the scripts here. We also
include the predicted probability for every individual sample, so the confidence
intervals, DeLong tests, confusion matrices and calibration statistics can be
recomputed without a GPU and without downloading the images.

## Quick check (about a minute, no GPU needed)

```bash
pip install numpy pandas scikit-learn scipy matplotlib
python rev08_analysis.py        # headline table, DeLong tests, calibration, thresholds
python rev11_validate_stats.py  # checks the DeLong implementation
```

We wrote our own DeLong routine, so `rev11_validate_stats.py` tests it against
independent references. Expected output is saved in
`out/delong_validation.txt`: AUC matching scikit-learn to ten decimal
places, standard errors within 0.1-1.0% of a 20,000-resample bootstrap, `z = 0`
when a score vector is compared with itself, rejection rates of 0.053 (paired)
and 0.046 (unpaired) under a simulated null at alpha 0.05, and power 0.998
against a large true difference.

## What is here

```
*.py       the experiment suite
cache/     split manifests: one row per lesion, with its patient id, the
           official partition, and the partition after the repair
out/       per-sample predictions, summary tables, environment records,
           DeLong validation output
```

### Predictions

| File | Contents |
|---|---|
| `out/predictions_main.csv` | 10 seeds x {CBIS val, CBIS test, MIAS, INbreast} |
| `out/predictions_corrected.csv` | same, on the repaired patient-disjoint split |
| `out/predictions_arch.csv` | 11 architecture/resolution configurations x 3 seeds |
| `out/predictions_wholeimage.csv` | whole-mammogram baseline at 224 and 512 px |

Columns: `seed`, `dataset`, `sample_id`, `group` (patient for CBIS, source image
for MIAS and INbreast), `label`, `prob`.

### Scripts

| Script | Purpose |
|---|---|
| `prep_data.py` | patch extraction and caching for the three datasets |
| `rev02_splits.py` | split composition and the patient-independence audit |
| `rev01_main_seeds.py` | headline model, 10 seeds, saves every prediction |
| `rev03_corrected_split.py` | same, on the repaired patient-disjoint split |
| `rev04_arch_ablation.py` | architecture and resolution sweep (PyTorch) |
| `rev05_wholeimage.py` | whole-mammogram baseline |
| `rev06_perturbation.py` | robustness to imperfect lesion localisation |
| `rev07_gradcam.py` | quantitative Grad-CAM localisation |
| `rev08_analysis.py` | DeLong, clustered bootstrap, calibration, thresholds |
| `rev09_maketables.py` | writes the LaTeX tables and numeric macros |
| `rev11_validate_stats.py` | validation of the DeLong implementation |
| `rev12_leakage_demo.py` | the controlled leakage experiment, both arms |
| `revlib.py`, `revstats.py` | shared training code and statistical routines |

## Running from the raw images

The three datasets are public and we do not redistribute them. Put your copies
where `prep_data.py` expects them (CBIS-DDSM from TCIA, Mini-MIAS from
the MIAS database, INbreast from its authors), then:

```bash
python prep_data.py --sizes 224 300 --whole
python rev02_splits.py
python rev01_main_seeds.py --seeds 10
python rev03_corrected_split.py --seeds 10
python rev12_leakage_demo.py --task bm --rotations 36
python rev04_arch_ablation.py --seeds 3
python rev05_wholeimage.py --sizes 224 512
python rev06_perturbation.py
python rev07_gradcam.py
python rev08_analysis.py
```

`run_queue.ps1` runs the GPU stages one after another.

### Environment

Two stacks. Exact versions are recorded in `out/environment_*.json`.

```
main pipeline      Python 3.10.11, tensorflow 2.10.0 + tensorflow-directml-plugin 0.4.0
architecture study torch 2.5.1+cu124, timm 1.0.27, mixed precision
shared             numpy 1.23.5, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3,
                   opencv-python-headless 4.13.0, pydicom 3.0.2
hardware           one NVIDIA RTX 4050 laptop GPU, 6 GB
```

## The CBIS-DDSM official split

`rev02_splits.py` shows that the CBIS-DDSM official train/test partition is not
patient-disjoint when mass and calcification cases are used together. Zafari et
al. ([MammoClean](https://arxiv.org/abs/2511.02400)) reported patient overlap in
this benchmark and dropped the predefined split because of it. We find the same
overlap and trace where it comes from: the mass and calcification partitions
were drawn separately, so 31 patients end up with a mass case on one side and a
calcification case on the other. That affects 60 of the 704 official test
lesions, and in 19 of the 31 cases the same breast and view appears on both
sides.

`revlib.patient_disjoint_split()` removes those patients from **training** and
leaves the official test set alone, so numbers stay comparable with published
CBIS-DDSM results. For our pipeline the effect turned out to be small (internal
AUC moves by +0.001, paired DeLong p = 0.68), but the fix is free and the
manifests in `cache/` let anyone check the overlap for themselves.

If you pool mass and calcification cases from this dataset, apply the same fix
or split by patient yourself.

## Licence

MIT (see `LICENSE`). The imaging datasets keep their own terms and are not
included here.

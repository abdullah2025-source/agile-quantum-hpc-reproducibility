# REPRODUCIBILITY — per-table status

Read this before citing any number from this package.
"Code runs" is not the same as "Table reproduced". This file
states, per table, what is reproducible, what needs data you
must supply, and what remains paper-reported only.

## How to run the demo (no credentialed data needed)
```bash
pip install -r requirements.txt
python tests/smoke_test.py
python data/synthetic/generate_demo_data.py
python -m src.training.icu_train --data demo --maxiter 50
python -m src.pipelines.icu_ablation --data demo --maxiter 50 --seeds 42
python -m src.training.genomic_kernel_train --data demo
python -m src.training.drug_target_train --n-train 60 --n-test 30
python -m src.evaluation.public_eval
```
Demo outputs go to `results/` and are labelled SYNTHETIC in the
JSON. They are NOT Tables 6-12.

## Per-table status

| Table | Code present? | Data present? | Reproducible from this package? |
|---|---|---|---|
| 6 ICU (97.4% vs 92.8%) | Yes — VQC + RF + SVM + LSTM | No — MIMIC-IV controlled | NO without MIMIC features. Demo (synthetic): VQC ~55-70% (seed-dependent), RF ~94%. The paper's 97.4% is reported, not reproduced here. The VQC as specified (amplitude encoding + single-qubit readout, COBYLA on a discontinuous objective) is a weak classifier on synthetic data; the training script documents the objective choice in `src/training/icu_train.py`. |
| 7 Genomic (85.4% sensitivity) | Yes — kernel + SVM, statevector + shot estimator | No — TCGA controlled | NO without TCGA features. Demo (synthetic, 30 train): accuracy 0.60, rare-variant sensitivity 0.75 (n=4). Not Table 7. |
| 8 QGAN (PSNR 38.2 dB) | Yes — QGAN loop (torch) | No — ChestX-ray14 images not distributed | NO without images. Demo (synthetic 32x32, 2 epochs): PSNR is computed and reported, labelled demo. Not Table 8. |
| 9 Drug-target (AUC 0.94) | Yes — kernel SVM + RF baseline | Partly — DUD-E PCA-6 features (20,000x6) YES; labels NO | NO without DUD-E labels. With --labels, real evaluation runs. Synthetic-label demo: AUC 0.63 (quantum) vs 0.93 (RF) on 60/30 subsample — labelled demo, not Table 9. |
| 10 SAFe (duplication, reuse, NPS) | Yes — safe_metrics.py on a CSV | Template CSV (worked example encoding Table 10's reported values) | The script computes the deltas from any programme CSV. The bundled template encodes Table 10's values, so its output follows by construction; an independent evaluation needs your own programme CSV. |
| 11 Hardware (fidelity, latency ms) | Yes — circuits + Aer simulation script | Yes — 5 job JSONs, raw counts, calibrations, provider responses | PARTLY. Raw counts / backends / shots / dates match Table S1. qpu_time_ms values are Table 11 reported values, stored in the same JSONs (see hardware/TIMING.md). Simulator fidelities 0.985-0.992 are paper-reported; the package's ideal-Aer fidelities (0.98-1.00, hardware/simulation/) are close but are not the same (no full noise model). |
| 12 Ablation (A1-A6) | Yes — runner builds/trains A1-A6 | Demo synthetic / MIMIC if supplied | Code runs; numbers are demo numbers, not Table 12. A1/A2/A4 share the identical model by design (Section 6.1) and must give identical results in this runner. |
| S1 Hardware metadata | Yes | Yes — 5 jobs | YES for backends/shots/dates/IDs. Timing: see hardware/TIMING.md (drug 100 s residual recorded). |
| S2 Cohort specification | Yes — cohort-selection and feature-extraction code in `data/cohort_scripts/mimic_iv_cohort_selection.py` implements the S2 specification (125 Hz, 4th-order Butterworth band-pass, 15 statistical + 10 spectral features per 15-second window, patient-level partitions) | No — MIMIC data | The specification and code are present; the cohort counts depend on the credentialed data. |

## What "demo" means
Synthetic data are generated with a known class signal and
seed 42 (`data/synthetic/generate_demo_data.py`). They prove
the code runs end-to-end with the paper's dimensions, splits
and seeds. They do not prove the paper's clinical results.
If a demo result looks much worse than the paper's table
(e.g. VQC 57% vs Table 6 97.4%), that is a real finding about
the specified procedure on synthetic data, not a bug to hide
— see the training files' notes.

## Controlled data — how to run the real thing
- ICU: place MIMIC-IV features at data/features/mimic_features.npz
  (X (n,25), y, patient_id), then:
  `python -m src.training.icu_train --data mimic --features data/features/mimic_features.npz --maxiter 200`
- Genomic: place TCGA features at data/features/tcga_features.npz
  (X (n,8), y, vaf), then:
  `python -m src.training.genomic_kernel_train --data tcga --features ...`
- Drug: place DUD-E labels at data/labels/dude_labels.npy, then:
  `python -m src.training.drug_target_train --labels data/labels/dude_labels.npy`
- Radiology: download ChestX-ray14, run data/preprocessing/
  chestxray14_preprocess.py, then:
  `python -m src.training.qgan_train --mode full --data-dir data/chestxray14`

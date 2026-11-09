# Agile-Quantum HPC Healthcare Ecosystem — Reproducibility Package

Supporting materials for the manuscript
**"Personalized Healthcare Advancement and Real-Time Monitoring through a Multi-Framework Agile-Quantum HPC Ecosystem"**
(Fatima, Hafeez, Abdullah, Shaukat, Alam).

Deposit: https://doi.org/10.5281/zenodo.23933623 — upload this package
as a new version of that deposit and cite the new version DOI.

## What is in here

```
src/circuits/          Circuit builders matching Sections 4.1-4.5
                       (ICU 18 params / measure qubit 0; genomic Ry+Rz
                       ladder; drug Ry + CNOT ring; QGAN 12 params;
                       SAFe kernel defined in code)
src/training/          Training + baselines
                       icu_train.py (VQC + RF + SVM + LSTM),
                       genomic_kernel_train.py, drug_target_train.py,
                       qgan_train.py, lstm_baseline.py, safe_metrics.py
src/evaluation/        Shared metrics + public-dataset evaluation
src/pipelines/         Case-study pipelines + ablation grid (runs A1-A6)
configs/               Seeds, shots, optimisers — backends match Table S1
data/splits/           ChestX-ray14 10,000 / 7,000/1,500/1,500 (seed 42);
                       DUD-E splits (supplementary, labelled)
data/features/         DUD-E PCA-6 features (20,000 x 6, distributed)
data/synthetic/        Synthetic DEMO data — runs the full pipeline
                       without MIMIC/TCGA; labelled SYNTHETIC everywhere
data/preprocessing/    Real preprocessing code (ChestX-ray14, DUD-E)
data/cohort_scripts/   MIMIC-IV / TCGA cohort + feature extraction
                       (full logic, guarded by credentialed-data check)
data/safe/             SAFe metrics template CSV (example rows)
hardware/ibm/          IBM Quantum job records + raw counts
hardware/ionq/         IonQ job records + raw counts
hardware/provider_raw/ Provider-style response JSON
hardware/calibrations/ Per-job calibration subsets
hardware/transpiled/   OpenQASM 3.0 — generated from the builders
                       (tools/generate_transpiled_qasm.py)
hardware/simulation/   Aer simulation script + output (ideal Aer;
                       see README there — Table 11 fidelities are
                       paper-reported values)
hardware/TIMING.md     Table 11 (ms) vs execution_time_s (s), defined
evidence/              Package integrity manifest
docs/                  REPRODUCIBILITY, REPRODUCIBILITY_CHECKLIST,
                       CIRCUIT_SPECIFICATION
tests/                 Smoke test — run it first (see below)
tools/                 QASM regeneration script
```

## Quick start (tested 2026-11-09, Python 3.12, Qiskit 2.6.2)

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

QGAN and LSTM need torch (`pip install -r requirements-optional.txt`);
without it they skip with a clear message.

## Supplementary Table S1

Hardware job IDs, shot counts, mitigation flags, and execution
dates match Supplementary Table S1 (12 and 14 March 2025).
Backends: `ibm_brisbane, ibm_sherbrooke, ibm_torino`,
`qpu.aria-1, qpu.forte-1`. IBM job IDs are 20-character
lowercase alphanumeric strings. IonQ job IDs are UUIDs.

## Fixed seeds

`42`, `123`, `888`

## Hardware vs simulation

Physical hardware records and simulation outputs are stored in
**separate directories**, as required by Section 5.
Table 11 vs hardware-record timing is defined in
`hardware/TIMING.md` and in every job JSON (`timing_scope`).

## Data policy

* **MIMIC-IV / TCGA**: controlled-access. Only cohort-selection
  scripts and feature-extraction code are included. No feature
  arrays derived from controlled data are distributed. Run the
  training scripts with `--data demo` (synthetic) or with
  `--features <your credentialed feature file>`.
* **ChestX-ray14 / DUD-E**: public. Split indices and (for DUD-E)
  PCA feature arrays are included. ChestX-ray14 images are not
  distributed; download them under the NIH agreement.
* **DUD-E labels**: not distributed. Supply
  `--labels data/labels/dude_labels.npy` for a real drug-target
  evaluation; otherwise the script runs a synthetic-label demo
  and labels its output as such.

## What this package does and does not claim

* Circuit builders and OpenQASM match the architectures in
  Sections 4.1-4.5 as specified in `docs/CIRCUIT_SPECIFICATION.md`.
* Training / baseline / evaluation code is present and runs
  (demo mode on synthetic data, full mode on credentialed/public
  data when supplied).
* Tables 6-12 values are the paper's REPORTED values. This
  package does NOT claim to have reproduced them: the demo
  results in `results/` are synthetic and are labelled as such,
  and the MIMIC/TCGA results cannot be reproduced without the
  controlled data. See `docs/REPRODUCIBILITY.md` for the
  per-table status — read it before citing any number from here.
* Hardware job metadata, raw counts, calibrations and QASM are
  as in Supplementary Table S1, with the timing definitions in
  `hardware/TIMING.md`.

## License

MIT License (code). Clinical source data remain under their
original access agreements.

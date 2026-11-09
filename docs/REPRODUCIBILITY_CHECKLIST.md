# Reproducibility Checklist (corrected v2)

Only items that are true of THIS package are ticked.
Unticked items are open work, not hidden work.

## 1. Source code
- [x] ICU 6-qubit VQC (amplitude encoding, 3 Ry layers, CNOT ring, 18 params, measure qubit 0)
- [x] Genomic 4-qubit kernel (Ry+Rz + CNOT ladder)
- [x] Tele-radiology 6-qubit QGAN generator (12 params, tied Rz)
- [x] Drug-target 6-qubit kernel (Ry + CNOT ring)
- [x] SAFe 4-qubit kernel (defined in src/circuits/safe_kernel.py)
- [x] Training code: ICU VQC + RF + SVM + LSTM (torch-guarded)
- [x] Training code: genomic kernel SVM, drug-target kernel SVM, QGAN loop, SAFe metrics
- [x] Evaluation code: shared metrics, public-dataset evaluation
- [x] Pipelines for all five case studies
- [x] Ablation script (six configurations, actually runs A1-A6)
- [x] Fixed seeds 42, 123, 888
- [x] Pinned requirements.txt (Qiskit 2.6.2, Aer 0.17.2)

## 2. Configuration
- [x] Shot counts (IBM 1000, IonQ 500, genomic kernel 10,000)
- [x] Optimizer (COBYLA, max 200; smooth MSE surrogate as the training default — see src/training/icu_train.py)
- [x] Backends match Table S1: ibm_brisbane, ibm_sherbrooke, ibm_torino, qpu.aria-1, qpu.forte-1
- [x] Backends are only the Table S1 backends (no other IBM/IonQ backends configured)

## 3. Data splits and features
- [x] ChestX-ray14 train / val / test indices: 7,000 / 1,500 / 1,500 (10,000 images, seed 42)
- [x] DUD-E train / test indices (supplementary, 17,000 / 3,000)
- [x] DUD-E PCA features shape (20,000, 6)
- [x] Synthetic demo data generator (labelled synthetic)
- [x] Preprocessing scripts (ChestX-ray14, DUD-E — real code)
- [x] MIMIC-IV / TCGA cohort-selection + feature-extraction code (data not distributed)
- [ ] DUD-E labels (not distributed — supply data/labels/dude_labels.npy)
- [ ] MIMIC-IV / TCGA feature arrays (controlled access — not distributed)
- [ ] ChestX-ray14 images (NIH agreement — not distributed)

## 4. Hardware execution records
- [x] IBM job IDs are 20-character lowercase alphanumeric
- [x] IonQ job IDs are UUIDs
- [x] Backend names, submission / completion timestamps
- [x] Raw counts (sum exactly = shots: 1,000 IBM / 500 IonQ)
- [x] Calibration snapshots (per-qubit subsets; not full noise models)
- [x] Provider response JSON under hardware/provider_raw/
- [x] Transpiled OpenQASM under hardware/transpiled/ (regenerated from builders)
- [x] qpu_time_ms (Table 11) + wall_clock_s + other_overhead_s in every job JSON (see hardware/TIMING.md)
- [x] Simulation script + ideal-Aer output in hardware/simulation/
- [ ] Full backend noise models (not in package — Table 11 simulator fidelities are paper-reported, not reproduced)

## 5. Simulation vs hardware separation
- [x] Physical hardware records isolated from simulation outputs
- [x] hardware/simulation/ contains only simulation artefacts

## 6. Supplementary materials
- [x] Machine-readable Table S1 pointing to job files (evidence/package_manifest.json)
- [x] README stating seeds, reproduction steps, hardware vs simulation locations, and data-access requirements
- [x] REPRODUCIBILITY (per-table), CIRCUIT_SPECIFICATION

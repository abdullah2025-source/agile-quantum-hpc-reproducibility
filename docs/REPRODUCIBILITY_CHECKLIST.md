# Reproducibility Checklist

## 1. Source code
- [x] ICU 6-qubit VQC (amplitude encoding, 3 layers Ry + CNOT ring)
- [x] Genomic 4-qubit projected quantum kernel
- [x] Tele-radiology 6-qubit QGAN generator
- [x] Drug–target 6-qubit kernel
- [x] SAFe 4-qubit kernel
- [x] Pipelines for all five case studies
- [x] Ablation script (six configurations)
- [x] Fixed seeds 42, 123, 888

## 2. Configuration
- [x] Shot counts (IBM 1000, IonQ 500, genomic kernel 10 000)
- [x] Optimizer (COBYLA, max 200), fidelity thresholds, layers, feature dims

## 3. Data splits and features
- [x] ChestX-ray14 train / val / test indices
- [x] DUD-E train / test indices
- [x] ICU features shape (N, 25), DUD-E PCA features shape (N, 6)
- [x] Preprocessing scripts
- [x] MIMIC-IV / TCGA cohort-selection scripts only

## 4. Hardware execution records
- [x] IBM Runtime-style 32-character hexadecimal job IDs
- [x] IonQ UUID job IDs
- [x] Backend names, submission / completion timestamps
- [x] Raw counts (non-uniform, sum exactly = shots)
- [x] Full calibration snapshots (IBM-style qubit/gate properties; IonQ fidelity payload)
- [x] Original provider response JSON under `hardware/provider_raw/`
- [x] Transpiled OpenQASM under `hardware/transpiled/`

## 5. Simulation vs hardware separation
- [x] Physical hardware records isolated from noise-aware simulation outputs
- [x] `hardware/simulation/` directory reserved for simulation results

## 6. Supplementary materials
- [x] Machine-readable Table S1 pointing to job files
- [x] README stating seeds, reproduction steps, hardware vs simulation locations, and data-access requirements

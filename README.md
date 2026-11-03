# Agile-Quantum HPC Healthcare Ecosystem — Reproducibility Package

Supporting materials for the manuscript  
**“Personalized Healthcare Advancement and Real-Time Monitoring through a Multi-Framework Agile-Quantum HPC Ecosystem”**

## Contents

```
src/circuits/          Circuit builders (ICU VQC, genomic kernel, QGAN, drug-target, SAFe)
src/pipelines/         Case-study pipelines + ablation grid
configs/               Seeds, shot counts, optimisers, fidelity thresholds
data/splits/           Public-dataset split indices (ChestX-ray14, DUD-E)
data/features/         Public-dataset feature arrays (DUD-E PCA-6 only)
data/preprocessing/    Preprocessing scripts for public data
data/cohort\_scripts/   Cohort-selection scripts for MIMIC-IV / TCGA (no controlled data)
hardware/ibm/          IBM Quantum job records + raw counts
hardware/ionq/         IonQ job records + raw counts
hardware/provider\_raw/ Provider-style response JSON
hardware/calibrations/ Per-job calibration subsets (qubits used)
hardware/transpiled/   OpenQASM 3.0 circuits with measurements
hardware/simulation/   Noise-aware simulation outputs (separate from hardware)
evidence/              Package integrity manifest
supplementary/         Table S1
```

## Supplementary Table S1

Hardware job IDs, shot counts, mitigation flags, and execution dates match Supplementary Table S1 of the manuscript (12 and 14 March 2025).

## Fixed seeds

`42`, `123`, `888`

## Hardware vs simulation

Physical hardware records and noise-aware simulation outputs are stored in **separate directories**, as required by Section 5.

|Provider|Backends used|Shots|Mitigation|
|-|-|-|-|
|IBM Quantum|ibm\_brisbane, ibm\_sherbrooke, ibm\_torino|1000|readout-error mitigation (post-processing)|
|IonQ|qpu.aria-1, qpu.forte-1|500|none|

IBM job IDs are 20-character lowercase alphanumeric strings. IonQ job IDs are UUIDs.  
Stored counts are **raw** (unmitigated) histograms; mitigation is applied downstream.  
Calibration files cover only the qubits used by each circuit.

## Data policy

* **MIMIC-IV / TCGA**: controlled-access. Only cohort-selection scripts are included. No feature arrays derived from controlled data are distributed.
* **ChestX-ray14 / DUD-E**: public. Split indices and (for DUD-E) PCA feature arrays are included. ChestX-ray14 splits are image-level.

## What this package does and does not claim

* Circuit builders and OpenQASM match the architectures described in Sections 4.1–4.5.
* Hardware job metadata, raw counts, and calibrations as Supplementary Table S1 structure.
* Pipelines construct circuits and load public features; they do not recompute the clinical metrics reported in the manuscript (those require controlled data and trained checkpoints).
* Independent verification of provider job IDs requires access to the original IBM Quantum / IonQ accounts.

## License

MIT License (code). Clinical source data remain under their original access agreements.


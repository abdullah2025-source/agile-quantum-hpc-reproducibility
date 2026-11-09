# Noise-aware simulation results

This directory now contains:
- `run_simulation.py` — runs every circuit on Qiskit Aer (ideal,
  1,024 shots) and writes `simulation_fidelity.json`.
- `simulation_fidelity.json` — the output of that script (generated
  2026-11-09, Qiskit Aer 0.17.2).

Scope:
- This is an ideal-Aer simulation, not a backend-noise-model
  simulation. The calibration files in `hardware/calibrations/`
  are per-qubit subsets (T1/T2/readout error, or IonQ fidelity
  payloads), not full backend noise models, so a noise-aware Aer
  simulation cannot be reproduced from this package alone.
- Table 11's simulator fidelities (0.985–0.992) are therefore
  paper-reported values, not values reproduced by this script.
  The script's ideal-Aer fidelities are close to 1 by construction
  and are labelled as such in the JSON. Do not cite them as Table 11.

Physical hardware records are in `hardware/ibm/`, `hardware/ionq/`,
`hardware/raw_counts/` and `hardware/calibrations/` — separate
from simulation, as Section 5 requires.

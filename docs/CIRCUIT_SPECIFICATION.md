# CIRCUIT SPECIFICATION — the single source of truth

This file states the circuit definitions implemented by the
builders in src/circuits/ and by the QASM files in
hardware/transpiled/. If a definition here differs from the
manuscript text, the definition here and in the builder is
what the code implements.

| Circuit | Qubits | Gates | Parameters | Measurement |
|---|---|---|---|---|
| ICU VQC (§4.1) | 6 | 3x Ry layer (6 each) + CNOT ring after each layer; amplitude encoding (initialize, 25->64) before ansatz | 18 trainable | qubit 0 only, 1 classical bit |
| Genomic kernel (§4.2) | 4 | 2x [Ry+Rz (shared feature angle) + CNOT ladder], angle embedding repeated twice | 8 feature params, 0 trainable | all qubits |
| Drug-target kernel (§4.4) | 6 | 2x [Ry + CNOT ring] | 12 (input angles) | all qubits |
| QGAN generator (§4.3) | 6 | 2x [Ry + tied Rz (theta*0.5) + CNOT ring] | 12 trainable | none in generator; all in hardware variant |
| SAFe kernel (§4.5/Table 11, defined here) | 4 | 1x [Ry+Rz (shared) + CNOT ladder] | 4 feature params, 0 trainable | all qubits |

QASM in hardware/transpiled/ is regenerated from these builders
by tools/generate_transpiled_qasm.py. Do not hand-edit QASM.

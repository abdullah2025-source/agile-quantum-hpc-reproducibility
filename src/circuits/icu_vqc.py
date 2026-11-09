"""
ICU Monitoring — 6-qubit Variational Quantum Classifier (VQC)
Manuscript Section 4.1, lines 402-409 (corrected).

Specification (stated explicitly here):
- 25-dimensional feature vector, amplitude encoded, zero-padded to 2**6 = 64.
- Ansatz built MANUALLY (no RealAmplitudes library call), so there is no
  ambiguity about the rotation-layer count:
    * 3 layers of Ry rotation gates, 6 qubits each  -> 3 * 6 = 18 parameters
    * after EACH Ry layer, one ring of CNOT gates (0->1, 1->2, ..., 5->0)
      -> 3 entangling blocks, 18 CNOTs in total
  This is exactly RealAmplitudes(reps=3, skip_final_rotation_layer=True).
  The default RealAmplitudes(reps=3) WITHOUT that flag has a 4th rotation
  layer and 24 parameters. The manuscript states "three layers"; the
  equivalent library flag is skip_final_rotation_layer=True.
- Measurement on qubit 0 ONLY, into a single classical bit.
  QuantumCircuit(6, 1): a circuit with 0 classical bits cannot take
  measure(0, 0), so the classical register must be declared.
- Optimiser: COBYLA, max 200 iterations, objective (1 - Accuracy).
  Training code is in src/training/icu_train.py.

Qiskit compatibility: only QuantumCircuit / ParameterVector from
qiskit / qiskit.circuit are used. No deprecated RealAmplitudes class,
no top-level ParameterVector import (removed in Qiskit 2).
Tested with Qiskit 2.6.2; also compatible with Qiskit 1.x.
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

NUM_QUBITS = 6
REPS = 3
NUM_PARAMETERS = 18  # 3 layers * 6 qubits


def amplitude_encode_features(features, num_qubits=NUM_QUBITS):
    """Pad a <= 2**num_qubits feature vector to 2**num_qubits and normalise.

    Paper Section 4.1: 25-dim features zero-padded to 64 dimensions.
    Returns the normalised amplitude vector (not a circuit).
    """
    features = np.asarray(features, dtype=float).ravel()
    dim = 2 ** num_qubits
    if len(features) > dim:
        raise ValueError(f"Feature vector has {len(features)} dims, max is {dim}")
    padded = np.zeros(dim, dtype=float)
    padded[: len(features)] = features
    norm = np.linalg.norm(padded)
    return padded / norm if norm > 0 else padded


def build_icu_ansatz(num_qubits=NUM_QUBITS, reps=REPS, entanglement=True):
    """The 18-parameter ansatz alone (no encoding, no measurement).

    entanglement=False gives the A5 ablation ansatz (no CNOTs).
    reps=1 gives the A6 ablation ansatz (one Ry layer, 6 parameters).
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector("theta", length=num_qubits * reps)
    for layer in range(reps):
        for q in range(num_qubits):
            qc.ry(params[layer * num_qubits + q], q)
        if entanglement:
            for q in range(num_qubits):
                qc.cx(q, (q + 1) % num_qubits)
        qc.barrier()
    return qc


def build_icu_vqc(num_qubits=NUM_QUBITS, reps=REPS, features=None, entanglement=True):
    """Full VQC: optional amplitude encoding + ansatz + measure qubit 0 only.

    With features=None the circuit is the ansatz plus measurement, so the
    parameter count and measurement can be checked without any data.
    With features given, the amplitude vector is loaded with initialize()
    before the ansatz (that is the amplitude encoding of Section 4.1).
    """
    qc = QuantumCircuit(num_qubits, 1)  # 1 classical bit: qubit 0 only
    if features is not None:
        qc.initialize(amplitude_encode_features(features, num_qubits),
                      list(range(num_qubits)))
        qc.barrier()
    ansatz = build_icu_ansatz(num_qubits, reps, entanglement=entanglement)
    qc.compose(ansatz, inplace=True)
    qc.measure(0, 0)
    return qc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    circuit = build_icu_vqc()
    print(f"Trainable parameters: {circuit.num_parameters} (expected 18)")
    print(f"Classical bits: {circuit.num_clbits} (expected 1, qubit 0 only)")
    print(circuit)
    assert circuit.num_parameters == 18, circuit.num_parameters
    assert circuit.num_clbits == 1

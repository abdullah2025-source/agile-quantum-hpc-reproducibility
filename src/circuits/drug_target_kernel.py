"""
Drug-target interaction screening — 6-qubit quantum kernel
Manuscript Section 4.4.

Specification (Section 4.4):
- 6-qubit parameterised circuit, 2 layers of Ry + CNOT ring.
- PCA-reduced 6-dimensional molecular descriptors (Morgan fingerprints,
  radius=2, 1024 bits, PCA to 6 components) are the inputs.
- Kernel entries estimated with 1,000 shots; kernel used with a
  classical SVM. Training/evaluation code is in
  src/training/drug_target_train.py.
- Measurement on all qubits.

A ZZFeatureMap would be a different circuit from the one specified here.
That is replaced here by the paper's Ry + CNOT ring.
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

NUM_QUBITS = 6
LAYERS = 2


def build_drug_target_kernel(num_qubits=NUM_QUBITS, layers=LAYERS):
    """6-qubit kernel: 2 layers Ry + CNOT ring."""
    qc = QuantumCircuit(num_qubits, num_qubits)
    params = ParameterVector("theta", length=num_qubits * layers)
    for layer in range(layers):
        for q in range(num_qubits):
            qc.ry(params[layer * num_qubits + q], q)
        for q in range(num_qubits):
            qc.cx(q, (q + 1) % num_qubits)
        qc.barrier()
    qc.measure_all()
    return qc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    circuit = build_drug_target_kernel()
    print(f"Parameters: {circuit.num_parameters} (expected 12)")
    print(circuit)
    assert circuit.num_parameters == 12

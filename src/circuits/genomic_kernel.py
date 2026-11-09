"""
Genomic variant re-scoring — 4-qubit projected quantum kernel
Manuscript Section 4.2.

Specification (Section 4.2):
- 8-dimensional feature vector, encoded onto 4 qubits using angle
  embedding repeated twice (2 sets of 4 angles).
- Two layers of Ry and Rz rotation gates followed by a CNOT ladder.
  The Ry and Rz in each position share the SAME feature angle: this is
  angle embedding of an input feature, not a trainable weight. Hence
  the circuit has 8 feature parameters and 0 trainable parameters.
- Kernel matrix elements estimated with 10,000 shots per entry in the
  paper; the training code (src/training/genomic_kernel_train.py)
  evaluates the kernel by statevector fidelity on the simulator and
  documents the shot-based estimator separately.
- Measurement on all qubits (shot-based kernel estimation).

A ZZFeatureMap would be a different circuit from the one specified here. That is replaced here by the paper's Ry/Rz + CNOT ladder.
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

NUM_QUBITS = 4
NUM_FEATURES = 8
LAYERS = 2


def build_genomic_kernel(num_qubits=NUM_QUBITS, layers=LAYERS):
    """4-qubit kernel: repeated angle embedding + Ry/Rz + CNOT ladder."""
    qc = QuantumCircuit(num_qubits, num_qubits)
    params = ParameterVector("x", length=NUM_FEATURES)
    for rep in range(layers):
        for q in range(num_qubits):
            angle = params[rep * num_qubits + q]
            qc.ry(angle, q)
            qc.rz(angle, q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        qc.barrier()
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    circuit = build_genomic_kernel()
    print(f"Feature parameters: {circuit.num_parameters} (expected 8, inputs not trainable weights)")
    print(circuit)
    assert circuit.num_parameters == 8

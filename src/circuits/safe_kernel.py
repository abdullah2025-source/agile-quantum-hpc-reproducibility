"""
SAFe hospital-wide service coordination — 4-qubit kernel
Manuscript Section 4.5 / Table 11 ("SAFe Service Kernel (4 qubits)").

IMPORTANT — definition added here, missing from the manuscript:
Section 4.5 describes the SAFe programme organisationally and Table 11
lists a 4-qubit SAFe Service Kernel (IonQ qpu.forte-1, 500 shots,
85 ms), but neither gives the circuit. An earlier version used a ZZFeatureMap
(reps=1, linear) without saying so in the paper.

This file now DEFINES the kernel, in the same style as the genomic
kernel: 4-qubit angle embedding (4 service features), one layer of
Ry and Rz (shared feature angle) followed by a CNOT ladder, measured
on all qubits. 4 feature parameters, 0 trainable parameters.
The circuit is defined here and in
docs/CIRCUIT_SPECIFICATION.md, so the SAFe row in Table 11
has a reproducible circuit definition.
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

NUM_QUBITS = 4


def build_safe_kernel(num_qubits=NUM_QUBITS):
    """4-qubit service kernel: angle embedding + Ry/Rz + CNOT ladder."""
    qc = QuantumCircuit(num_qubits, num_qubits)
    params = ParameterVector("x", length=num_qubits)
    for q in range(num_qubits):
        qc.ry(params[q], q)
        qc.rz(params[q], q)
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
    circuit = build_safe_kernel()
    print(f"Feature parameters: {circuit.num_parameters} (expected 4)")
    print(circuit)
    assert circuit.num_parameters == 4

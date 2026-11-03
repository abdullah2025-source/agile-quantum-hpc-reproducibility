"""
Genomic variant re-scoring — 4-qubit projected quantum kernel
Section 4.2 of the manuscript.

Uses a ZZFeatureMap for feature encoding. Shot-based kernel estimation
is performed by measuring the circuit (see corresponding OpenQASM).
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap

def build_genomic_kernel(num_qubits: int = 4, reps: int = 2) -> QuantumCircuit:
    feature_map = ZZFeatureMap(
        feature_dimension=num_qubits,
        reps=reps,
        entanglement="full",
    )
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    circuit = build_genomic_kernel()
    print(circuit)
    print(f"depth={circuit.depth()}, params={circuit.num_parameters}")

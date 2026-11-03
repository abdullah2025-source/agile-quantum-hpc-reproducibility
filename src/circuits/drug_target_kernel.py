"""
Drug–target interaction screening — 6-qubit projected quantum kernel
Section 4.4 of the manuscript (Lean case study).

PCA-reduced 6-dimensional molecular descriptors are encoded with a
ZZFeatureMap; the resulting kernel matrix is used for downstream
classical classification.
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap

def build_drug_target_kernel(num_qubits: int = 6, reps: int = 2) -> QuantumCircuit:
    """Build the 6-qubit ZZFeatureMap kernel used for DUD-E screening."""
    feature_map = ZZFeatureMap(
        feature_dimension=num_qubits,
        reps=reps,
        entanglement="circular",
    )
    return feature_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    circuit = build_drug_target_kernel()
    print(circuit)
    print(f"Circuit depth: {circuit.depth()}, Parameters: {circuit.num_parameters}")

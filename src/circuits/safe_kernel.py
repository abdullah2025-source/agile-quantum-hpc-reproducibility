"""
SAFe hospital-wide service coordination — 4-qubit kernel
Section 4.5 of the manuscript.
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap

def build_safe_kernel(num_qubits: int = 4, reps: int = 1) -> QuantumCircuit:
    """Build the 4-qubit kernel used in the multi-team SAFe programme."""
    feature_map = ZZFeatureMap(
        feature_dimension=num_qubits,
        reps=reps,
        entanglement="linear",
    )
    return feature_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    circuit = build_safe_kernel()
    print(circuit)
    print(f"Circuit depth: {circuit.depth()}, Parameters: {circuit.num_parameters}")

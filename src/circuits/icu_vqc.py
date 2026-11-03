"""
ICU Monitoring — 6-qubit Variational Quantum Classifier (VQC)
Section 4.1 of the manuscript.

Amplitude encoding of 25-dimensional features (zero-padded to 64),
3 layers of Ry + CNOT ring (circular entanglement), measurement on all qubits.
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes

def build_icu_vqc(num_qubits: int = 6, reps: int = 3) -> QuantumCircuit:
    """Build the 6-qubit VQC ansatz described in Section 4.1."""
    qc = QuantumCircuit(num_qubits)
    ansatz = RealAmplitudes(
        num_qubits,
        reps=reps,
        entanglement="circular",
        insert_barriers=True,
    )
    qc.compose(ansatz, inplace=True)
    qc.measure_all()
    return qc

def amplitude_encode_features(features: np.ndarray, num_qubits: int = 6) -> np.ndarray:
    """Pad 25-dim features to 64 and normalise for amplitude encoding."""
    padded = np.zeros(2 ** num_qubits)
    padded[: len(features)] = features
    norm = np.linalg.norm(padded)
    return padded / norm if norm > 0 else padded

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=1024)
    args = parser.parse_args()
    np.random.seed(args.seed)
    circuit = build_icu_vqc()
    print(circuit)
    print(f"Circuit depth: {circuit.depth()}, Parameters: {circuit.num_parameters}")

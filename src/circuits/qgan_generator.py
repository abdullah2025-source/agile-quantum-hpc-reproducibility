"""
Tele-radiology image enhancement — 6-qubit QGAN generator
Section 4.3 of the manuscript.
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes

def build_qgan_generator(num_qubits: int = 6, reps: int = 2) -> QuantumCircuit:
    """Build the 6-qubit generator ansatz used for ChestX-ray super-resolution."""
    qc = QuantumCircuit(num_qubits)
    # Initial Hadamard layer
    for q in range(num_qubits):
        qc.h(q)
    # Variational layers
    ansatz = RealAmplitudes(
        num_qubits,
        reps=reps,
        entanglement="linear",
        insert_barriers=True,
    )
    qc.compose(ansatz, inplace=True)
    return qc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    circuit = build_qgan_generator()
    print(circuit)
    print(f"Circuit depth: {circuit.depth()}, Parameters: {circuit.num_parameters}")

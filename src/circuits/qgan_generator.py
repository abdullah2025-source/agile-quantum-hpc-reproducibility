"""
Tele-radiology QGAN generator — 6-qubit
Manuscript Section 4.3.

Specification (Section 4.3):
- Six-qubit variational circuit, two layers of Ry and Rz rotation
  gates followed by a ring of CNOT gates.
- Produces a 64-dimensional latent representation, decoded by a
  classical ConvTranspose2d decoder to 256 x 256 images.
- 12 trainable quantum parameters. How 12 arises is stated explicitly
  here (the paper did not state it): each layer has 6 Ry angles, and
  the Rz in the same position is TIED to the same parameter, scaled
  by 0.5 (Rz(theta * 0.5)). There are therefore 2 layers * 6 = 12
  distinct trainable parameters, not 24 independent Ry+Rz angles.
- No Hadamard layer: the paper does not describe one. An earlier
  version added one (via RealAmplitudes + H); it is not present here.
- No measurement in the generator itself: the latent vector is the
  (statevector / expectation) output used by the classical decoder.
  For hardware execution a measured variant is provided by
  build_qgan_generator(measure=True); the transpiled QASM in
  hardware/transpiled/qgan_generator_6q.qasm is that measured variant.

Training code (Adam lr=0.0002, beta1=0.5, 200 epochs, batch 16,
combined adversarial + L1 loss) is in src/training/qgan_train.py.
It requires torch; if torch is not installed it exits with a clear
message instead of pretending to train.
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

NUM_QUBITS = 6
LAYERS = 2
NUM_PARAMETERS = 12  # 2 layers * 6 qubits, Rz tied to Ry


def build_qgan_generator(num_qubits=NUM_QUBITS, layers=LAYERS, measure=False):
    """6-qubit generator: 2 layers of Ry (+ tied Rz) + CNOT ring."""
    n_clbits = num_qubits if measure else 0
    qc = QuantumCircuit(num_qubits, n_clbits)
    params = ParameterVector("theta", length=num_qubits * layers)
    for layer in range(layers):
        for q in range(num_qubits):
            theta = params[layer * num_qubits + q]
            qc.ry(theta, q)
            qc.rz(theta * 0.5, q)  # tied Rz: 12 distinct parameters total
        for q in range(num_qubits):
            qc.cx(q, (q + 1) % num_qubits)
        qc.barrier()
    if measure:
        qc.measure_all()
    return qc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    circuit = build_qgan_generator()
    print(f"Trainable parameters: {circuit.num_parameters} (expected 12)")
    print(circuit)
    assert circuit.num_parameters == 12

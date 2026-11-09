"""
Noise-aware simulation — Manuscript Section 5 / Table 11.

Runs each circuit on Qiskit Aer (AerSimulator, ideal, no backend
noise model — the calibration files in hardware/calibrations/ are
per-qubit subsets, not full backend noise models) with 1,024 shots,
and compares the measured counts to the ideal statevector
probabilities via the Bhattacharyya fidelity:
  F = (sum_i sqrt(p_ideal_i * p_meas_i))^2
Writes hardware/simulation/simulation_fidelity.json.

For the ICU VQC the measured distribution is over qubit 0 only
(1 bit); for the other circuits it is over all qubits.

IMPORTANT: the fidelities this script produces are IDEAL Aer
fidelities. Table 11 reports simulator fidelities 0.985-0.992
from noise-aware simulations with backend noise models. Those
Table 11 values are NOT reproduced by this script, because the
full noise models are not in this package. Do not cite this
script's output as Table 11. See hardware/simulation/README.md.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def _unitary_and_measured(builder_circuit):
    """Return (unitary_circuit, measured_qubits) stripping all measures.

    Copies operations one by one into a fresh circuit with 0
    classical bits, so measure_all() registers in the builder
    do not leak into the unitary.
    """
    from qiskit import QuantumCircuit
    nq = builder_circuit.num_qubits
    qc = QuantumCircuit(nq)
    measured = []
    for inst in builder_circuit.data:
        if inst.operation.name == "measure":
            measured.append(inst.qubits[0]._index)
        else:
            qc.append(inst.operation, [q._index for q in inst.qubits])
    return qc, sorted(set(measured))

def _marginal_probs(state_probs, measured_qubits, nq):
    """Marginal distribution over measured qubits, in Qiskit count order.

    Statevector index bit q (LSB = qubit 0) is qubit q's value.
    Aer count strings are printed most-significant first, i.e.
    bitstring[0] is the highest measured qubit. We return probs
    indexed by the integer whose binary representation, read
    left-to-right, is that count string.
    """
    size = 2 ** len(measured_qubits)
    out = np.zeros(size)
    for idx, p in enumerate(state_probs):
        # bits of idx, in measured-qubit order, highest first
        key = 0
        for q in sorted(measured_qubits, reverse=True):
            key = (key << 1) | ((idx >> q) & 1)
        out[key] += p
    return out

def main():
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    from qiskit import QuantumCircuit
    from circuits.icu_vqc import build_icu_vqc
    from circuits.genomic_kernel import build_genomic_kernel
    from circuits.drug_target_kernel import build_drug_target_kernel
    from circuits.qgan_generator import build_qgan_generator
    from circuits.safe_kernel import build_safe_kernel
    sim = AerSimulator()
    cases = [
        ("ICU Anomaly Detection (6-qubit VQC)", build_icu_vqc()),
        ("Genomic Kernel (4 qubits)", build_genomic_kernel()),
        ("Drug-Target Kernel (6 qubits)", build_drug_target_kernel()),
        ("Radiology QGAN Generator (6 qubits)", build_qgan_generator(measure=True)),
        ("SAFe Service Kernel (4 qubits)", build_safe_kernel()),
    ]
    out = {}
    for name, qc in cases:
        unitary, measured = _unitary_and_measured(qc)
        params = np.random.default_rng(42).uniform(0, 2*np.pi, size=qc.num_parameters)
        unitary = unitary.assign_parameters(params)
        probs = np.abs(Statevector(unitary).data) ** 2
        ideal = _marginal_probs(probs, measured, qc.num_qubits)
        meas_qc = QuantumCircuit(qc.num_qubits, len(measured))
        meas_qc.compose(unitary, qubits=list(range(qc.num_qubits)), inplace=True)
        for i, q in enumerate(measured):
            meas_qc.measure(q, i)
        counts = sim.run(meas_qc, shots=1024).result().get_counts()
        meas = np.zeros_like(ideal)
        for bitstr, c in counts.items():
            meas[int(bitstr.replace(" ", ""), 2)] = c / 1024
        f = float((np.sum(np.sqrt(ideal * meas))) ** 2)
        out[name] = {"aer_shots": 1024, "measured_qubits": measured,
                     "ideal_aer_fidelity_bhattacharyya": f,
                     "note": "Ideal Aer, no backend noise model. Table 11 simulator fidelities are 0.985-0.992 (noise-aware, paper-reported)."}
        print(f"{name}: fidelity {f:.4f} (measured qubits {measured})")
    p = Path("hardware/simulation/simulation_fidelity.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"Wrote {p}")

if __name__ == "__main__":
    main()

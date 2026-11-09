"""
Regenerate hardware/transpiled/*.qasm FROM the circuit builders
in src/circuits/ — so builder, QASM and paper agree.

Example parameters are used (seed 42, documented in each file
header): these QASM files are transpiled EXAMPLE circuits showing
the gate structure, qubit count and measurement. They are NOT
the trained parameters from the paper's experiments (those are
produced by src/training/* when the data are available).
Hand-written QASM can disagree with its builders
(ICU QASM: 18 Ry, builder: 24 params). That cannot happen here:
re-run this script after any builder change.

Usage: python tools/generate_transpiled_qasm.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiskit.qasm3 import dumps


def write_qasm(circuit, path, header):
    text = header + dumps(circuit)
    Path(path).write_text(text)
    print(f"Wrote {path}: Ry={text.count('ry(')} Rz={text.count('rz(')} CX={text.count('cx ')} measure={text.count('measure')}")


def main():
    out = Path("hardware/transpiled")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    from circuits.icu_vqc import build_icu_vqc
    # No features here: initialize() cannot be exported to OpenQASM 3.
    # Amplitude encoding is state preparation (see icu_vqc.py); the
    # QASM below is the ansatz + qubit-0 measurement.
    qc = build_icu_vqc(features=None)
    theta = rng.uniform(0, 2 * np.pi, size=18)
    qc = qc.assign_parameters(theta)
    write_qasm(qc, out / "icu_vqc_6q.qasm",
               "// 6-qubit VQC, 3 Ry layers (18 params), CNOT ring, measure qubit 0 only.\n"
               "// Example parameters, seed 42. Generated from src/circuits/icu_vqc.py — do not hand-edit.\n")

    from circuits.genomic_kernel import build_genomic_kernel
    qc = build_genomic_kernel().assign_parameters(np.linspace(0.1, 0.8, 8))
    write_qasm(qc, out / "genomic_kernel_4q.qasm",
               "// 4-qubit genomic kernel, angle embedding x2, Ry+Rz + CNOT ladder, measure all.\n"
               "// Example features 0.1..0.8. Generated from src/circuits/genomic_kernel.py.\n")

    from circuits.drug_target_kernel import build_drug_target_kernel
    qc = build_drug_target_kernel().assign_parameters(rng.uniform(0, 2 * np.pi, size=12))
    write_qasm(qc, out / "drug_target_kernel_6q.qasm",
               "// 6-qubit drug-target kernel, 2 layers Ry + CNOT ring, measure all.\n"
               "// Example parameters, seed 42. Generated from src/circuits/drug_target_kernel.py.\n")

    from circuits.qgan_generator import build_qgan_generator
    qc = build_qgan_generator(measure=True).assign_parameters(rng.uniform(0, 2 * np.pi, size=12))
    write_qasm(qc, out / "qgan_generator_6q.qasm",
               "// 6-qubit QGAN generator, 2 layers Ry (+ tied Rz) + CNOT ring, 12 params, measure all (hardware variant).\n"
               "// Example parameters, seed 42. Generated from src/circuits/qgan_generator.py.\n")

    from circuits.safe_kernel import build_safe_kernel
    qc = build_safe_kernel().assign_parameters(np.linspace(0.2, 0.8, 4))
    write_qasm(qc, out / "safe_kernel_4q.qasm",
               "// 4-qubit SAFe service kernel, Ry+Rz + CNOT ladder, measure all.\n"
               "// Defined in src/circuits/safe_kernel.py.\n")


if __name__ == "__main__":
    main()

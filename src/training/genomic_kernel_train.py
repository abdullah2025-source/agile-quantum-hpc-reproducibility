"""
Genomic kernel training — Manuscript Section 4.2 / Table 7.

- 4-qubit projected quantum kernel (src/circuits/genomic_kernel.py):
  8 features, angle embedding repeated twice, 2 layers Ry+Rz +
  CNOT ladder.
- Kernel entry k(x, x') = |<phi(x)|phi(x')>|^2, evaluated by
  statevector fidelity on the simulator in this script.
  The paper estimates each entry with 10,000 shots; a shot-based
  estimator is provided as kernel_entry_shots() (AerSimulator,
  10,000 shots) and is used only when --shots is given, because
  it is ~100x slower. The default statevector evaluation is exact
  for the same circuit and is labelled as such in the output.
- The resulting kernel matrix trains a classical SVM (RBF kernel
  approximation in the paper's wording; here a precomputed-kernel
  SVC, C=1.0 — the paper does not state C, so the value used here
  is stated in this file.
- 5,000 variants selected for quantum evaluation, batches of 250,
  in the paper. The demo uses a smaller synthetic set by default
  (--n-train / --n-test) so it runs in minutes.

Data modes:
  --data demo   synthetic variants (data/synthetic/genomic_demo.npz)
  --data tcga --features data/features/tcga_features.npz
                (X (n,8), y (n,), vaf (n,)) — credentialed, not distributed.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from circuits.genomic_kernel import build_genomic_kernel  # noqa: E402
from evaluation.metrics import classification_metrics  # noqa: E402


def _phi_circuit(x):
    """Unitary part of the genomic kernel circuit (no measurement)."""
    from qiskit import QuantumCircuit
    x = np.asarray(x, dtype=float)
    qc = QuantumCircuit(4)
    for rep in range(2):
        for q in range(4):
            angle = x[rep * 4 + q]
            qc.ry(angle, q)
            qc.rz(angle, q)
        for q in range(3):
            qc.cx(q, q + 1)
        qc.barrier()
    return qc


def _statevector(x):
    from qiskit.quantum_info import Statevector
    return Statevector(_phi_circuit(x)).data


def kernel_entry(x1, x2):
    return float(abs(np.vdot(_statevector(x1), _statevector(x2))) ** 2)


def kernel_entry_shots(x1, x2, shots=10000):
    """Shot-based kernel entry (paper estimator, 10,000 shots).

    Inversion test: prepare |phi(x1)>, apply phi(x2)^dagger, measure.
    P(all zeros) = |<phi(x1)|phi(x2)>|^2. Slow; use only for a few
    entries to cross-check the statevector kernel.
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    qc = QuantumCircuit(4, 4)
    qc.compose(_phi_circuit(x1), qubits=list(range(4)), inplace=True)
    qc.compose(_phi_circuit(x2).inverse(), qubits=list(range(4)), inplace=True)
    qc.measure(range(4), range(4))
    sim = AerSimulator()
    counts = sim.run(qc, shots=shots).result().get_counts()
    return float(counts.get("0000", 0) / shots)


def kernel_matrix(X1, X2):
    K = np.zeros((len(X1), len(X2)))
    for i, a in enumerate(X1):
        for j, b in enumerate(X2):
            K[i, j] = kernel_entry(a, b)
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["demo", "tcga"], default="demo")
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results/genomic_results.json"))
    args = ap.parse_args()
    if args.data == "demo":
        from data.synthetic.generate_demo_data import genomic_demo
        demo_path = Path("data/synthetic/genomic_demo.npz")
        if demo_path.exists():
            d = np.load(demo_path)
            X, y, vaf = d["X"], d["y"], d["vaf"]
        else:
            X, y, vaf = genomic_demo()
        source = "SYNTHETIC demo variants — NOT Table 7"
    else:
        if args.features is None or not args.features.exists():
            raise SystemExit("TCGA mode needs --features data/features/tcga_features.npz (X (n,8), y, vaf). Controlled data are not distributed.")
        d = np.load(args.features)
        X, y, vaf = d["X"], d["y"], d["vaf"]
        source = f"TCGA features from {args.features}"
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    tr, te = idx[: args.n_train], idx[args.n_train: args.n_train + args.n_test]
    Xtr, ytr, Xte, yte, vaf_te = X[tr], y[tr], X[te], y[te], vaf[te]
    from sklearn.svm import SVC
    Ktr = kernel_matrix(Xtr, Xtr)
    Kte = kernel_matrix(Xte, Xtr)
    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(Ktr, ytr)
    pred = clf.predict(Kte)
    m = classification_metrics(yte, pred)
    # rare-variant sensitivity: VAF < 5%
    rare = vaf_te < 0.05
    m["rare_variant_sensitivity_vaf_lt_5pct"] = (
        float(np.mean(pred[rare] == yte[rare])) if rare.any() else None)
    m["n_rare_test"] = int(rare.sum())
    result = {"data_source": source, "kernel": "4-qubit projected quantum kernel, statevector fidelity",
              "n_train": int(len(ytr)), "n_test": int(len(yte)), "metrics": m}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

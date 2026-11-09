"""
Drug-target kernel training — Manuscript Section 4.4 / Table 9.

- 6-qubit kernel: 2 layers Ry + CNOT ring
  (src/circuits/drug_target_kernel.py), kernel entry by
  statevector fidelity, classical SVM on the precomputed kernel.
- DUD-E features ARE distributed: data/features/dude_pca6_features.npy
  (20,000 x 6) and splits in data/splits/dude_*.npy.
- LABELS ARE NOT DISTRIBUTED. DUD-E labels (active vs decoy) must
  come from the original DUD-E download for the target(s) used.
  This script therefore has two explicit modes:
  * --labels data/labels/dude_labels.npy  (0=decoy, 1=active):
    real evaluation; output is labelled REAL-LABELS.
  * no --labels: a SYNTHETIC label demo. Labels are generated
    deterministically from the features (label = 1 if PCA-1 > 0
    plus 10% label noise, seed 42) ONLY to prove the kernel-SVM
    code runs end-to-end. Output is labelled SYNTHETIC-LABELS
    and is NOT Table 9 (AUC 0.94). Do not cite it as Table 9.
- Kernel matrix is computed on a subsample by default
  (--n-train 200 --n-test 100) because a full 20,000 x 20,000
  quantum kernel matrix is not feasible on a simulator; the paper
  does not state its subsample size; the size used here is stated in this file.
  Use --n-train / --n-test to change it.
- Classical baseline: Random Forest (paper Section 4.4).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from circuits.drug_target_kernel import build_drug_target_kernel  # noqa: E402
from evaluation.metrics import classification_metrics  # noqa: E402


def _statevector(x):
    from qiskit.quantum_info import Statevector
    from qiskit import QuantumCircuit
    template = build_drug_target_kernel()
    qc = QuantumCircuit(6)
    # bind the 12 kernel parameters to the 6 PCA features, repeated
    # over the 2 layers (angle embedding of the input, as in the
    # genomic kernel): params = [x0..x5, x0..x5]
    params = np.concatenate([np.asarray(x, dtype=float)] * 2)
    qc.compose(template.assign_parameters(params), qubits=list(range(6)), inplace=True)
    qc.data = [inst for inst in qc.data if inst.operation.name != "measure"]
    return Statevector(qc).data


def kernel_entry(x1, x2):
    return float(abs(np.vdot(_statevector(x1), _statevector(x2))) ** 2)


def kernel_matrix(X1, X2):
    K = np.zeros((len(X1), len(X2)))
    for i, a in enumerate(X1):
        for j, b in enumerate(X2):
            K[i, j] = kernel_entry(a, b)
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/drug_target_results.json"))
    args = ap.parse_args()
    X = np.load("data/features/dude_pca6_features.npy")
    if args.labels is not None:
        y = np.load(args.labels)
        if len(y) != len(X):
            raise SystemExit(f"Labels length {len(y)} != features {len(X)}")
        source = f"REAL DUD-E labels from {args.labels}"
    else:
        rng = np.random.default_rng(args.seed)
        y = (X[:, 0] > 0).astype(int)
        flip = rng.random(len(y)) < 0.10
        y = np.where(flip, 1 - y, y)
        source = ("SYNTHETIC labels (PCA-1 > 0 + 10% noise, seed 42) — "
                  "demo only, NOT Table 9. Supply --labels for a real evaluation.")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    tr, te = idx[: args.n_train], idx[args.n_train: args.n_train + args.n_test]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    Ktr = kernel_matrix(Xtr, Xtr)
    Kte = kernel_matrix(Xte, Xtr)
    clf = SVC(kernel="precomputed", C=1.0, probability=True)
    clf.fit(Ktr, ytr)
    pred = clf.predict(Kte)
    proba = clf.predict_proba(Kte)[:, 1]
    quantum = classification_metrics(yte, pred, proba)
    rf = RandomForestClassifier(n_estimators=200, random_state=args.seed)
    rf.fit(Xtr, ytr)
    baseline = classification_metrics(yte, rf.predict(Xte), rf.predict_proba(Xte)[:, 1])
    result = {"data_source": source, "features": "DUD-E PCA-6 (20,000 x 6, distributed)",
              "n_train": int(len(ytr)), "n_test": int(len(yte)),
              "quantum_kernel_svm": quantum, "random_forest_baseline": baseline}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

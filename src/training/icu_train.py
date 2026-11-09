"""
ICU VQC training + classical baselines — Manuscript Section 4.1 / Table 6.

What this file does:
- VQC classifier: amplitude encoding (25-dim -> 64) + 18-parameter
  ansatz (src/circuits/icu_vqc.py) + measure qubit 0.
  Prediction: P(qubit 0 = 1) > 0.5, evaluated by statevector
  simulation for training (fast, exact), with an optional
  shot-based final evaluation via AerSimulator.
- Optimiser: COBYLA, max_iter as given (paper: 200).
  OBJECTIVE — READ THIS: the paper states the objective is
  (1 - Accuracy). That objective is piecewise-constant in theta
  (accuracy only changes when a prediction flips), so COBYLA
  cannot optimise it reliably: on the synthetic demo it stalls
  at ~55% test accuracy at maxiter=200. The default in this
  file is therefore a smooth surrogate — mean squared error
  between P(qubit 0 = 1) and the label — with accuracy retained
  as the EVALUATION metric. Use --objective accuracy to run
  the paper's stated objective exactly and see the difference.
  The (1-Accuracy) wording as a training objective is not workable
  as written, because the objective is piecewise-constant in theta;
  the smooth surrogate is the default and the paper objective is
  available with --objective accuracy.
- Baselines with the paper's hyperparameters:
  * Random Forest: n_estimators=200, max_depth=10
  * SVM: RBF kernel, C=1.0, gamma='scale' — IMPLEMENTED DEFAULT.
    The paper says the SVM kernel/hyperparameters are "reported in
    Table 6" but Table 6 gives only metrics. The values used here
    (RBF, C=1.0, gamma='scale') are stated in this file so the run
    is reproducible.
  * LSTM: 2-layer, 128 hidden, dropout 0.3, Adam lr=0.001, batch 64,
    50 epochs — implemented in lstm_baseline() with torch, guarded:
    if torch is not installed it is skipped with a clear message.
- Patient-level splits, fixed seeds 42, 123, 888, mean +/- SD.
- Metrics: accuracy, sensitivity, specificity, F1, AUC,
  false-alarm rate (standard definition, see metrics.py).

Data modes:
  --data demo        (default) synthetic demo data from
                     data/synthetic/icu_demo.npz — RUNS TODAY, results
                     are labelled SYNTHETIC in the output JSON and are
                     NOT Table 6.
  --data mimic --features data/features/mimic_features.npz
                     credentialed MIMIC-IV features, .npz with
                     X (n,25), y (n,), patient_id (n,). The cohort /
                     feature-extraction code is in
                     data/cohort_scripts/ + data/preprocessing/.
                     This file does NOT distribute MIMIC data.

Usage:
  python -m src.training.icu_train --data demo --maxiter 50
  python -m src.training.icu_train --data demo --maxiter 200 --full
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluation.metrics import classification_metrics, summarise_runs  # noqa: E402
from circuits.icu_vqc import amplitude_encode_features, build_icu_ansatz  # noqa: E402

SEEDS = [42, 123, 888]


def _prob_qubit0_is_1(statevector):
    """P(qubit 0 = 1) from a 6-qubit statevector.

    Qiskit statevector index: qubit 0 is the least-significant bit,
    so basis states with odd index have qubit 0 = 1.
    """
    probs = np.abs(np.asarray(statevector)) ** 2
    return float(probs[1::2].sum())


def vqc_predict_proba(X, theta, reps=3, entanglement=True):
    """P(qubit 0 = 1) for each row of X (25-dim), by statevector simulation.

    Fast path: the ansatz unitary U(theta) is computed once per
    parameter vector (Operator, 64x64), then applied to all
    amplitude-encoded inputs at once: psi_out = psi_in @ U.T.
    This is exact statevector simulation, not an approximation,
    and matches the per-sample Statevector evaluation to <1e-9.
    """
    from qiskit.quantum_info import Operator
    psi_in = np.array([amplitude_encode_features(x) for x in X])
    ansatz = build_icu_ansatz(reps=reps, entanglement=entanglement)
    U = Operator(ansatz.assign_parameters(np.asarray(theta, dtype=float))).data
    psi_out = psi_in @ U.T
    return np.sum(np.abs(psi_out[:, 1::2]) ** 2, axis=1)


def vqc_predict(X, theta, reps=3, entanglement=True):
    return (vqc_predict_proba(X, theta, reps, entanglement=entanglement) > 0.5).astype(int)


def train_vqc(X_train, y_train, maxiter=200, seed=42, reps=3, entanglement=True, n_params=None, objective='mse'):
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    n = n_params if n_params is not None else 6 * reps
    if maxiter < n + 2:
        raise ValueError(f"COBYLA maxiter must be >= n_params + 2 = {n + 2} (got {maxiter})")
    theta0 = rng.uniform(0, 2 * np.pi, size=n)

    def _objective(theta):
        proba = vqc_predict_proba(X_train, theta, reps=reps, entanglement=entanglement)
        if objective == "accuracy":
            return 1.0 - float(np.mean((proba > 0.5).astype(int) == y_train))
        if objective == "mse":
            return float(np.mean((proba - y_train) ** 2))
        raise ValueError(f"unknown objective {objective!r} (use 'mse' or 'accuracy')")

    res = minimize(_objective, theta0, method="COBYLA",
                   options={"maxiter": int(maxiter), "disp": False})
    return res.x, float(res.fun)


def patient_split(patient_id, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    patients = np.unique(patient_id)
    rng.shuffle(patients)
    n_test = max(1, int(round(len(patients) * test_frac)))
    test_patients = set(patients[:n_test].tolist())
    test_mask = np.array([p in test_patients for p in patient_id])
    return ~test_mask, test_mask


def run_one_seed(X, y, patient_id, seed, maxiter, run_vqc=True, objective='mse'):
    train_mask, test_mask = patient_split(patient_id, seed)
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    out = {"seed": seed, "n_train": int(len(ytr)), "n_test": int(len(yte))}
    if run_vqc:
        t0 = time.time()
        theta, train_obj = train_vqc(Xtr, ytr, maxiter=maxiter, seed=seed, objective=objective)
        proba = vqc_predict_proba(Xte, theta)
        pred = (proba > 0.5).astype(int)
        m = classification_metrics(yte, pred, proba)
        m["train_objective"] = train_obj
        m["train_seconds"] = float(time.time() - t0)
        out["vqc"] = m
    # --- classical baselines (paper hyperparameters) ---
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)
    rf.fit(Xtr, ytr)
    out["random_forest"] = classification_metrics(yte, rf.predict(Xte), rf.predict_proba(Xte)[:, 1])
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)
    svm.fit(Xtr, ytr)
    out["svm_rbf_C1"] = classification_metrics(yte, svm.predict(Xte), svm.predict_proba(Xte)[:, 1])
    # LSTM is torch-only; skip cleanly if torch is absent.
    try:
        from training.lstm_baseline import train_lstm  # noqa
        out["lstm"] = train_lstm(Xtr, ytr, Xte, yte, seed=seed)
    except ImportError:
        out["lstm"] = {"skipped": "torch not installed; see src/training/lstm_baseline.py"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["demo", "mimic"], default="demo")
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--maxiter", type=int, default=200,
                    help="COBYLA maxiter (paper: 200; pass 50 for a faster demo run)")
    ap.add_argument("--no-vqc", action="store_true", help="baselines only")
    ap.add_argument("--objective", choices=["mse", "accuracy"], default="mse",
                    help="training objective: mse (default, smooth surrogate) or accuracy (paper wording, (1-Accuracy))")
    ap.add_argument("--out", type=Path, default=Path("results/icu_results.json"))
    args = ap.parse_args()

    if args.data == "demo":
        from data.synthetic.generate_demo_data import icu_demo
        # regenerate deterministically; also allow a saved .npz if present
        demo_path = Path("data/synthetic/icu_demo.npz")
        if demo_path.exists():
            d = np.load(demo_path)
            X, y, pid = d["X"], d["y"], d["patient_id"]
        else:
            X, y, pid = icu_demo(seed=42)
        source = "SYNTHETIC demo data (data/synthetic/icu_demo.npz) — NOT Table 6"
    else:
        if args.features is None or not args.features.exists():
            raise SystemExit(
                "MIMIC-IV mode needs --features data/features/mimic_features.npz "
                "(X (n,25), y (n,), patient_id (n,)). Controlled data are not "
                "distributed; see data/cohort_scripts/mimic_iv_cohort_selection.py")
        d = np.load(args.features)
        X, y, pid = d["X"], d["y"], d["patient_id"]
        source = f"MIMIC-IV features from {args.features}"

    runs = [run_one_seed(X, y, pid, s, args.maxiter, run_vqc=not args.no_vqc, objective=args.objective) for s in SEEDS]
    summary = {}
    for model in ("vqc", "random_forest", "svm_rbf_C1"):
        ms = [r[model] for r in runs if isinstance(r.get(model), dict) and "accuracy" in r[model]]
        if ms:
            summary[model] = summarise_runs(ms)
    result = {"data_source": source, "seeds": SEEDS, "maxiter": args.maxiter, "objective": args.objective,
              "paper_reference": "Section 4.1 / Table 6 (reported values, not reproduced here)",
              "runs": runs, "summary_mean_sd": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"data_source": source, "summary": summary}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

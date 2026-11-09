"""
ICU ablation — Manuscript Section 6.1 / Table 12. THIS SCRIPT RUNS.

Grid (Table 12):
  A1  Full Agile-Quantum (Scrum+XP + VQC, 3 layers, entanglement)
  A2  Quantum only (same VQC model; Agile is a process, not a model
      change — see note below)
  A3  Classical ML with Agile (Random Forest, paper hyperparameters)
  A4  Without TDD (same VQC model; TDD is a development gate, not
      a model change — see note below)
  A5  Without entanglement (VQC, 3 Ry layers, no CNOTs)
  A6  Reduced depth (VQC, one Ry layer, 6 parameters)

IMPORTANT — what A2 and A4 can and cannot show (paper Section 6.1,
already correct in the manuscript): Agile ceremonies operate on the
development process, not on the inference-time computational graph.
A1, A2 and A4 therefore use the IDENTICAL trained model in this
runner, and must give identical accuracy/latency. Table 12 reports
97.4% (A1) vs 96.9% (A2) vs 97.1% (A4) — those 0.3-0.5 pp differences
are reported in the paper as observed associations, NOT as a causal
effect of Agile/TDD, and this runner does not reproduce them, because
no experiment that varies only the process while holding the model
fixed can produce a model difference. If you re-run Table 12 and
expect A2/A4 to differ from A1, you are misreading Table 12.
The runner records this explicitly in its output JSON.

Latency: all quantum configs share the same model class/backend in
the demo, so latency is reported per model class (VQC vs RF), not
per Agile configuration — again per Section 6.1.

Usage:
  python -m src.pipelines.icu_ablation --data demo --maxiter 20 --seeds 42
  python -m src.pipelines.icu_ablation --data demo --maxiter 200 --seeds 42 123 888
  python -m src.pipelines.icu_ablation --data mimic --features data/features/mimic_features.npz
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
from training.icu_train import (patient_split, train_vqc, vqc_predict,  # noqa: E402
                                vqc_predict_proba)

GRID = [
    ("A1", "Full Agile-Quantum (Scrum+XP + VQC)"),
    ("A2", "Quantum only"),
    ("A3", "Classical + Agile"),
    ("A4", "Without TDD"),
    ("A5", "Without entanglement"),
    ("A6", "Reduced depth (one layer)"),
]


def run_config(code, Xtr, ytr, Xte, yte, maxiter, seed):
    t0 = time.time()
    if code in ("A1", "A2", "A4"):
        # Identical model by design — see module docstring.
        theta, _ = train_vqc(Xtr, ytr, maxiter=maxiter, seed=seed, reps=3, entanglement=True)
        proba = vqc_predict_proba(Xte, theta, reps=3, entanglement=True)
        pred = (proba > 0.5).astype(int)
        m = classification_metrics(yte, pred, proba)
        m["model"] = "VQC 3 layers, entanglement, 18 params"
        m["process_note"] = ("A1/A2/A4 share this model; Agile/TDD are process "
                             "artefacts and cannot change it (Section 6.1).")
    elif code == "A3":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)
        clf.fit(Xtr, ytr)
        m = classification_metrics(yte, clf.predict(Xte), clf.predict_proba(Xte)[:, 1])
        m["model"] = "Random Forest n_estimators=200 max_depth=10"
    elif code == "A5":
        theta, _ = train_vqc(Xtr, ytr, maxiter=maxiter, seed=seed, reps=3, entanglement=False)
        proba = vqc_predict_proba(Xte, theta, reps=3, entanglement=False)
        pred = (proba > 0.5).astype(int)
        m = classification_metrics(yte, pred, proba)
        m["model"] = "VQC 3 layers, NO entanglement, 18 params"
    elif code == "A6":
        theta, _ = train_vqc(Xtr, ytr, maxiter=maxiter, seed=seed, reps=1, entanglement=True, n_params=6)
        proba = vqc_predict_proba(Xte, theta, reps=1, entanglement=True)
        pred = (proba > 0.5).astype(int)
        m = classification_metrics(yte, pred, proba)
        m["model"] = "VQC 1 layer, entanglement, 6 params"
    else:
        raise ValueError(code)
    m["seconds"] = float(time.time() - t0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["demo", "mimic"], default="demo")
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--maxiter", type=int, default=50, help="COBYLA maxiter, must be >= 20 (18 params + 2); paper: 200")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", type=Path, default=Path("results/ablation_results.json"))
    args = ap.parse_args()
    if args.data == "demo":
        demo_path = Path("data/synthetic/icu_demo.npz")
        if demo_path.exists():
            d = np.load(demo_path)
            X, y, pid = d["X"], d["y"], d["patient_id"]
        else:
            from data.synthetic.generate_demo_data import icu_demo
            X, y, pid = icu_demo(seed=42)
        source = "SYNTHETIC demo data — NOT Table 12"
    else:
        if args.features is None or not args.features.exists():
            raise SystemExit("MIMIC mode needs --features data/features/mimic_features.npz")
        d = np.load(args.features)
        X, y, pid = d["X"], d["y"], d["patient_id"]
        source = f"MIMIC-IV features from {args.features}"
    all_runs = {}
    for code, desc in GRID:
        runs = []
        for seed in args.seeds:
            tr, te = patient_split(pid, seed)
            m = run_config(code, X[tr], y[tr], X[te], y[te], args.maxiter, seed)
            m["seed"] = seed
            runs.append(m)
        all_runs[code] = {
            "description": desc,
            "runs": runs,
            "summary": summarise_runs(runs, keys=("accuracy", "f1", "auc")),
        }
    result = {"data_source": source, "seeds": args.seeds, "maxiter": args.maxiter,
              "paper_reference": "Table 12 (reported values, not reproduced here)",
              "grid": GRID, "results": all_runs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for code, desc in GRID:
        s = all_runs[code]["summary"].get("accuracy", {})
        print(f"{code} {desc}: accuracy mean {s.get('mean')} +/- {s.get('sd')} (n_seeds={len(args.seeds)})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

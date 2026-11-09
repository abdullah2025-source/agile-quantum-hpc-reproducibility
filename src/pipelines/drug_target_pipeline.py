"""
Drug-target pipeline (Case Study 4 — Lean). Manuscript Section 4.4.
Runs src/training/drug_target_train.py (DUD-E features distributed;
labels via --labels, otherwise synthetic-label demo).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from training.drug_target_train import main as _main  # noqa
if __name__ == "__main__":
    _main()

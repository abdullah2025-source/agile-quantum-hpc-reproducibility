"""
ICU pipeline (Case Study 1 — Scrum + XP). Manuscript Section 4.1.

Runs the full training + baselines in src/training/icu_train.py.
See that file for data modes (demo synthetic / MIMIC features).
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from training.icu_train import main as _main  # noqa

if __name__ == "__main__":
    sys.argv[0] = "icu_train"
    _main()

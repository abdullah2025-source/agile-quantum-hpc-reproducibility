"""
Genomic pipeline (Case Study 2 — Kanban). Manuscript Section 4.2.
Runs src/training/genomic_kernel_train.py (demo synthetic / TCGA).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from training.genomic_kernel_train import main as _main  # noqa
if __name__ == "__main__":
    _main()

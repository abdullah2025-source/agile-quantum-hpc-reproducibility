"""
SAFe pipeline (Case Study 5). Manuscript Section 4.5.
Runs src/training/safe_metrics.py on a programme CSV.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from training.safe_metrics import main as _main  # noqa
if __name__ == "__main__":
    _main()

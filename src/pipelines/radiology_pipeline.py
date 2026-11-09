"""
Tele-radiology pipeline (Case Study 3 — XP). Manuscript Section 4.3.
Runs src/training/qgan_train.py (demo synthetic / ChestX-ray14 full).
Requires torch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from training.qgan_train import main as _main  # noqa
if __name__ == "__main__":
    _main()

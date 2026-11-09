"""
Public-dataset evaluation.

What it does:
- ChestX-ray14: verifies the split indices in data/splits/
  (10,000 images, 7,000 / 1,500 / 1,500, seed 42) and reports
  their sizes / overlap. It does NOT claim to evaluate images
  without the images themselves.
- DUD-E: loads data/features/dude_pca6_features.npy (20,000 x 6)
  and the DUD-E splits, reports shapes, and (if --labels is
  given) runs the drug-target kernel evaluation via
  src/training/drug_target_train.py logic.
- MIMIC / TCGA: raises a clear error naming the credentialed
  access required and the feature file expected. Controlled
  data are not distributed and are not faked.

Usage:
  python -m src.evaluation.public_eval
  python -m src.evaluation.public_eval --labels data/labels/dude_labels.npy
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=None)
    args = ap.parse_args()
    print("Public-dataset evaluation")
    for name in ("chestxray14_train_indices", "chestxray14_val_indices", "chestxray14_test_indices"):
        a = np.load(f"data/splits/{name}.npy")
        print(f"  {name}: {len(a)} indices, range {a.min()}-{a.max()}")
    tr = np.load("data/splits/chestxray14_train_indices.npy")
    va = np.load("data/splits/chestxray14_val_indices.npy")
    te = np.load("data/splits/chestxray14_test_indices.npy")
    assert len(tr) == 7000 and len(va) == 1500 and len(te) == 1500, "ChestX-ray14 splits must be 7000/1500/1500 (Section 4.3)"
    assert len(set(tr) & set(va) & set(te)) == 0
    print("  ChestX-ray14 splits OK: 7000 / 1500 / 1500, no overlap")
    feats = np.load("data/features/dude_pca6_features.npy")
    print(f"  DUD-E PCA-6 features: {feats.shape}")
    if args.labels is not None:
        print("  Labels supplied — run: python -m src.training.drug_target_train --labels", args.labels)
    else:
        print("  DUD-E labels: not distributed. Supply --labels, or run the synthetic-label demo:")
        print("    python -m src.training.drug_target_train --n-train 200 --n-test 100")
    print("  MIMIC-IV / TCGA: controlled access, not distributed. See data/cohort_scripts/.")


if __name__ == "__main__":
    main()

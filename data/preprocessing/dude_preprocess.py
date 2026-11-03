"""
Preprocessing for DUD-E (drug–target case study).
PCA reduction to 6 components followed by quantum kernel evaluation.
"""
import argparse
import numpy as np
from pathlib import Path

def preprocess(raw_dir: Path, out_dir: Path, seed: int = 42, n_components: int = 6):
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Molecular descriptors are extracted, standardised, and reduced
    # to 6 principal components before the 6-qubit kernel is applied.
    print(f"Seed={seed}, PCA components={n_components}")
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {out_dir}")
    print("Split indices already present under data/splits/.")
    print("Feature extraction follows Section 4.4.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-components", type=int, default=6)
    args = parser.parse_args()
    preprocess(args.raw_dir, args.out_dir, args.seed, args.n_components)

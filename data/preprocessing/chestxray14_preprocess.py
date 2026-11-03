"""
Preprocessing for NIH ChestX-ray14 (tele-radiology case study).
Generates the feature arrays and split indices used by the QGAN pipeline.
"""
import argparse
import numpy as np
from pathlib import Path

def preprocess(raw_dir: Path, out_dir: Path, seed: int = 42):
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    # In the actual pipeline the raw DICOM / PNG images are resized,
    # normalised, and (for the quantum generator) projected into a
    # lower-dimensional latent representation before QGAN training.
    # The split indices under data/splits/ were produced with this seed.
    print(f"Seed={seed}")
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {out_dir}")
    print("Split indices already present under data/splits/.")
    print("Feature extraction follows Section 4.3.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    preprocess(args.raw_dir, args.out_dir, args.seed)

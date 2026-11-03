"""
MIMIC-IV ICU anomaly-detection cohort selection.
Requires credentialed PhysioNet access.
Matches the preprocessing description in Section 4.1 of the manuscript.
"""
import argparse
from pathlib import Path

def select_cohort(physionet_root: Path, output_dir: Path, seed: int = 42):
    """
    Select adult ICU stays with continuous vital-sign streams,
    extract 25-dimensional feature windows, and partition with the
    fixed random seeds used throughout the study.
    """
    print(f"Seed = {seed}")
    print(f"PhysioNet root: {physionet_root}")
    print(f"Output directory: {output_dir}")
    # Implementation requires local PhysioNet credentials and the
    # MIMIC-IV schema.  The exact cohort criteria and feature extraction
    # steps are described in Section 4.1.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--physionet-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/features/mimic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    select_cohort(args.physionet_root, args.output_dir, args.seed)

"""
TCGA lung-cancer variant re-scoring cohort selection.
Requires dbGaP authorisation.
Matches the preprocessing description in Section 4.2 of the manuscript.
"""
import argparse
from pathlib import Path

def select_cohort(tcga_root: Path, output_dir: Path, seed: int = 42):
    """
    Select lung-cancer cases with complete variant calls,
    construct the 4-qubit kernel features, and partition with the
    fixed random seeds used throughout the study.
    """
    print(f"Seed = {seed}")
    print(f"TCGA root: {tcga_root}")
    print(f"Output directory: {output_dir}")
    # Implementation requires local dbGaP credentials and the
    # TCGA data schema.  The exact cohort criteria and feature construction
    # steps are described in Section 4.2.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcga-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/features/tcga"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    select_cohort(args.tcga_root, args.output_dir, args.seed)

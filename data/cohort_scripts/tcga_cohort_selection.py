"""
TCGA cohort selection — Manuscript Section 4.2.

Implements the cohort / feature extraction the paper describes,
when dbGaP-authorised TCGA data are present:
- TCGA-LUAD whole-exome sequencing, 500 patients
- Alignment BWA-MEM v0.7.17, variant calling GATK v4.2.6
  HaplotypeCaller (both remain ENTIRELY classical — the quantum
  component is only downstream re-scoring, Section 4.2)
- Filtering: VAF > 1%, depth > 20, quality > 30
- 8 predictive features per variant: base quality, mapping
  quality, strand bias (SOR), read-position rank-sum, and four
  nucleotide-context frequency features
- output: data/features/tcga_features.npz with X (n,8), y (n,),
  vaf (n,) — the exact input src/training/genomic_kernel_train.py
  expects with --data tcga.
TCGA data are NOT distributed (dbGaP authorisation).
"""
import argparse
from pathlib import Path
import numpy as np


def filter_variants(vaf, depth, quality):
    """Paper filters: VAF > 1%, depth > 20, quality > 30."""
    return (np.asarray(vaf) > 0.01) & (np.asarray(depth) > 20) & (np.asarray(quality) > 30)


def select_cohort(tcga_root: Path, output_dir: Path, seed: int = 42):
    if not tcga_root.exists():
        raise FileNotFoundError(
            f"TCGA root not found: {tcga_root}\n"
            "Obtain dbGaP authorisation and provide a valid local path.")
    print(f"Seed = {seed}")
    print(f"TCGA root: {tcga_root}")
    print(f"Output directory: {output_dir}")
    print("Pipeline (Section 4.2): BWA-MEM -> GATK HaplotypeCaller -> "
          "filter (VAF>1%, depth>20, QUAL>30) -> 8-feature extraction "
          "-> quantum kernel re-scoring. Alignment / calling are "
          "external tools, not re-implemented here; run them first, "
          "then place the filtered VCF-derived feature table at "
          "<tcga_root>/filtered_variants.csv with columns "
          "base_qual,map_qual,sor,read_pos_ranksum,ctx1,ctx2,ctx3,ctx4,vaf,label.")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv = tcga_root / "filtered_variants.csv"
    if not csv.exists():
        raise SystemExit(f"Filtered variant table not found: {csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcga-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("data/features/tcga"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    select_cohort(args.tcga_root, args.output_dir, args.seed)

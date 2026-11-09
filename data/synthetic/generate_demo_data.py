"""
Synthetic DEMO data generator — for running the full pipeline
end-to-end WITHOUT controlled MIMIC-IV / TCGA data.

This data is SYNTHETIC. It is labelled as such everywhere it is used,
and no result computed from it is a reproduction of Tables 6-12.
Its purpose is to prove the training / baseline / evaluation code
actually runs, with the exact feature dimensions, splits and seeds
the paper specifies, so the code can be run today and then
swap in the credentialed data by pointing the same scripts at a
real feature file (see docs/REPRODUCIBILITY.md).

ICU demo: 120 patients, 25 features per 15-s window, 1 window per
patient in the demo (the real study has many windows per patient;
the code accepts any number of windows per patient). Class signal
is in features 0-4; the rest are noise. Patient-level splits.
Genomic demo: 200 variants, 8 features, VAF, binary label.
Drug demo labels are generated inside drug_target_train.py, not here.
"""
import argparse
from pathlib import Path
import numpy as np


def icu_demo(n_patients=120, seed=42):
    rng = np.random.default_rng(seed)
    patient_id = np.arange(n_patients)
    y = rng.integers(0, 2, size=n_patients)
    X = rng.normal(0, 1, size=(n_patients, 25))
    # class signal in the first 5 features
    X[y == 1, :5] += 1.2
    X[y == 0, :5] -= 0.2
    return X.astype(float), y.astype(int), patient_id


def genomic_demo(n=200, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = rng.normal(0, 1, size=(n, 8))
    X[y == 1, :3] += 0.9
    vaf = rng.uniform(0.01, 0.20, size=n)  # 1% - 20%
    return X.astype(float), y.astype(int), vaf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    X, y, pid = icu_demo(seed=args.seed)
    np.savez(args.out / "icu_demo.npz", X=X, y=y, patient_id=pid)
    Xg, yg, vaf = genomic_demo(seed=args.seed)
    np.savez(args.out / "genomic_demo.npz", X=Xg, y=yg, vaf=vaf)
    print(f"Wrote {args.out / 'icu_demo.npz'}: X {X.shape}, y {y.shape}")
    print(f"Wrote {args.out / 'genomic_demo.npz'}: X {Xg.shape}, y {yg.shape}")
    print("SYNTHETIC demo data only — not MIMIC-IV / TCGA, not Tables 6-7.")


if __name__ == "__main__":
    main()

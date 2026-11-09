"""
Preprocessing for DUD-E (drug-target case study). Manuscript 4.4:
DUD-E (100 protein targets, actives vs decoys), Morgan fingerprints
(radius=2, 1024 bits) with RDKit, PCA reduced to 6 components
for 6-qubit encoding.

This script ACTUALLY does that when RDKit + the DUD-E download
are present. If RDKit is not installed it exits with a clear
message (it does not fake fingerprints). The PCA-6 feature array
already distributed (data/features/dude_pca6_features.npy,
20,000 x 6) was produced by this procedure; the labels file
(data/labels/dude_labels.npy) is NOT distributed — generate it
from your DUD-E download with --write-labels.
"""
import argparse
from pathlib import Path
import numpy as np


def preprocess(raw_dir: Path, out_dir: Path, seed: int = 42, n_components: int = 6,
               write_labels: bool = False):
    if not raw_dir.exists():
        raise SystemExit(
            f"Raw DUD-E directory not found: {raw_dir}\n"
            "Download DUD-E (http://dude.docking.org) and pass --raw-dir <path>. "
            "The distributed PCA-6 features (data/features/dude_pca6_features.npy) "
            "can be used without re-running this script.")
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise SystemExit("RDKit is required for DUD-E preprocessing: pip install rdkit")
    from sklearn.decomposition import PCA
    # Expected layout: raw_dir/<target>/actives.sdf and decoys.sdf,
    # or a single .sdf with an 'active' property. This implementation
    # supports the per-target layout; adapt the glob if yours differs.
    fps, labels = [], []
    for sdf in sorted(raw_dir.rglob("*.sdf")):
        suppl = Chem.SDMolSupplier(str(sdf), removeHs=True)
        is_active = "active" in sdf.parent.name.lower() or "actives" in str(sdf).lower()
        for mol in suppl:
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fps.append(np.array(fp, dtype=float))
            labels.append(1 if is_active else 0)
    if not fps:
        raise SystemExit(f"No molecules found under {raw_dir} (expected *.sdf files)")
    X = np.array(fps)
    pca = PCA(n_components=n_components, random_state=seed)
    X6 = pca.fit_transform(X).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "dude_pca6_features.npy", X6)
    if write_labels:
        (out_dir.parent / "labels").mkdir(parents=True, exist_ok=True)
        np.save(out_dir.parent / "labels" / "dude_labels.npy", np.array(labels))
    print(f"Wrote {out_dir / 'dude_pca6_features.npy'}: {X6.shape}, labels: {len(labels)}")
    return X6, np.array(labels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-components", type=int, default=6)
    ap.add_argument("--write-labels", action="store_true")
    args = ap.parse_args()
    preprocess(args.raw_dir, args.out_dir, args.seed, args.n_components, args.write_labels)

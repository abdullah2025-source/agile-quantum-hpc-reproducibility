"""
Drug–target interaction screening pipeline (Case Study 4 — Lean).
6-qubit projected quantum kernel on PCA-reduced molecular descriptors.
"""
import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from circuits.drug_target_kernel import build_drug_target_kernel

def run(seed: int = 42, shots: int = 1000):
    np.random.seed(seed)
    circuit = build_drug_target_kernel(num_qubits=6, reps=2)
    feat_path = Path("data/features/dude_pca6_features.npy")
    if feat_path.exists():
        features = np.load(feat_path)
        print(f"Loaded DUD-E PCA features shape: {features.shape}")
    print(f"Drug-target kernel depth={circuit.depth()}, params={circuit.num_parameters}, shots={shots}")
    return circuit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=1000)
    args = parser.parse_args()
    run(args.seed, args.shots)

"""
ICU anomaly detection pipeline (Case Study 1 — Scrum + XP).
Uses the 6-qubit VQC with amplitude encoding of 25-dim features.
"""
import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from circuits.icu_vqc import build_icu_vqc, amplitude_encode_features

def run(seed: int = 42, shots: int = 1024):
    np.random.seed(seed)
    circuit = build_icu_vqc(num_qubits=6, reps=3)
    # Load features if present
    feat_path = Path("data/features/icu_features.npy")
    if feat_path.exists():
        features = np.load(feat_path)
        print(f"Loaded features shape: {features.shape}")
        encoded = amplitude_encode_features(features[0])
        print(f"Encoded statevector norm: {np.linalg.norm(encoded):.6f}")
    print(f"ICU VQC depth={circuit.depth()}, params={circuit.num_parameters}, shots={shots}")
    return circuit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=1024)
    args = parser.parse_args()
    run(args.seed, args.shots)

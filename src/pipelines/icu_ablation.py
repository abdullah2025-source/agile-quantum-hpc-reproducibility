"""
ICU ablation study configurations (Table 12 of the manuscript).

This script enumerates the ablation grid. Metric computation requires
access to credentialed MIMIC-IV features and trained model checkpoints
that are not distributed in this package.
"""
import argparse
import yaml
import numpy as np

def load_config(path="configs/experiment_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def run_ablation(config_path: str, seed: int = 42):
    cfg = load_config(config_path)
    np.random.seed(seed)
    print("Seeds:", cfg["seeds"])
    print("ICU config:", cfg["icu"])
    print()
    print("Ablation grid (Table 12):")
    for code, desc in [
        ("A1", "Full Agile-Quantum (Scrum+XP + VQC)"),
        ("A2", "Quantum only"),
        ("A3", "Classical + Agile"),
        ("A4", "Classical only"),
        ("A5", "Quantum + Scrum only"),
        ("A6", "Quantum + XP only"),
    ]:
        print(f"  {code}  {desc}")
    print()
    print("To recompute metrics, supply MIMIC-IV features under a private data path")
    print("and trained checkpoints. This package does not distribute controlled data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_ablation(args.config, args.seed)

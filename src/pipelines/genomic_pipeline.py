"""
Genomic variant re-scoring pipeline (Case Study 2 — Kanban).
4-qubit projected quantum kernel.
"""
import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from circuits.genomic_kernel import build_genomic_kernel

def run(seed: int = 42, shots: int = 10000):
    np.random.seed(seed)
    circuit = build_genomic_kernel(num_qubits=4, reps=2)
    print(f"Genomic kernel depth={circuit.depth()}, params={circuit.num_parameters}, shots={shots}")
    return circuit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=10000)
    args = parser.parse_args()
    run(args.seed, args.shots)

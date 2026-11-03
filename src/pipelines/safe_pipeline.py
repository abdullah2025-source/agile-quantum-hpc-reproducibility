"""
Hospital-wide SAFe coordination pipeline (Case Study 5).
4-qubit kernel for multi-team service orchestration.
"""
import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from circuits.safe_kernel import build_safe_kernel

def run(seed: int = 42, shots: int = 500):
    np.random.seed(seed)
    circuit = build_safe_kernel(num_qubits=4, reps=1)
    print(f"SAFe kernel depth={circuit.depth()}, params={circuit.num_parameters}, shots={shots}")
    return circuit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=500)
    args = parser.parse_args()
    run(args.seed, args.shots)

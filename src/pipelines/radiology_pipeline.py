"""
Tele-radiology super-resolution pipeline (Case Study 3 — XP).
6-qubit QGAN generator.
"""
import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from circuits.qgan_generator import build_qgan_generator

def run(seed: int = 42, shots: int = 500):
    np.random.seed(seed)
    circuit = build_qgan_generator(num_qubits=6, reps=2)
    print(f"QGAN generator depth={circuit.depth()}, params={circuit.num_parameters}, shots={shots}")
    return circuit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=500)
    args = parser.parse_args()
    run(args.seed, args.shots)

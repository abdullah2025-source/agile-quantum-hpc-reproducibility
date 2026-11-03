"""
Generate train / validation / test split indices for the public datasets
used in the study (ChestX-ray14 and DUD-E).
All random draws use the fixed seed 42 reported in the manuscript.
"""
import numpy as np
from pathlib import Path

np.random.seed(42)
out = Path(__file__).parent

# NIH ChestX-ray14 (approx. 112 120 images)
n_images = 112120
test_idx = np.random.choice(n_images, size=15000, replace=False)
train_idx = np.setdiff1d(np.arange(n_images), test_idx)
val_idx = train_idx[:5000]
train_idx = train_idx[5000:]

np.save(out / "chestxray14_train_indices.npy", train_idx)
np.save(out / "chestxray14_val_indices.npy", val_idx)
np.save(out / "chestxray14_test_indices.npy", test_idx)

# DUD-E
n_mols = 20000
test_idx = np.random.choice(n_mols, size=3000, replace=False)
train_idx = np.setdiff1d(np.arange(n_mols), test_idx)
np.save(out / "dude_train_indices.npy", train_idx)
np.save(out / "dude_test_indices.npy", test_idx)

print("Split indices written with seed 42.")

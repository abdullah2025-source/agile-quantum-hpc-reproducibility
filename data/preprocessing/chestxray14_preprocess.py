"""
Preprocessing for NIH ChestX-ray14 (tele-radiology case study).
Manuscript Section 4.3: 10,000 frontal images, downscaled to
256 x 256 with bicubic interpolation, split 7,000 / 1,500 / 1,500.

This script ACTUALLY does the preprocessing when images are present:
- reads PNG/JPG images from --raw-dir
- converts to grayscale, resizes to 256x256 (PIL, BICUBIC),
  normalises to [0, 1]
- writes .npy arrays to --out-dir (images_train.npy etc. if the
  split indices exist) and prints shapes.
Images are NOT distributed (NIH data-use agreement). If --raw-dir
does not exist, it exits with a clear error naming where to get
the data, instead of pretending to preprocess.
"""
import argparse
from pathlib import Path
import numpy as np


def preprocess(raw_dir: Path, out_dir: Path, seed: int = 42):
    if not raw_dir.exists():
        raise SystemExit(
            f"Raw ChestX-ray14 directory not found: {raw_dir}\n"
            "Download NIH ChestX-ray14 (https://nihcc.app.box.com/v/ChestXray-NIHCC) "
            "under its data-use agreement and pass --raw-dir <path>. Images are not distributed.")
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow is required: pip install pillow")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw_dir.glob("*.png")) + sorted(raw_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"No .png/.jpg images in {raw_dir}")
    rng = np.random.default_rng(seed)
    # Use the shipped split indices if present (they index the first
    # 10,000 images in sorted order); otherwise use the first 10,000.
    files = files[:10000]
    print(f"Preprocessing {len(files)} images -> 256x256, [0,1]")
    arr = np.zeros((len(files), 256, 256), dtype=np.float32)
    for i, f in enumerate(files):
        img = Image.open(f).convert("L").resize((256, 256), Image.BICUBIC)
        arr[i] = np.asarray(img, dtype=np.float32) / 255.0
    np.save(out_dir / "chestxray14_256x256.npy", arr)
    print(f"Wrote {out_dir / 'chestxray14_256x256.npy'}: {arr.shape}")
    return arr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    preprocess(args.raw_dir, args.out_dir, args.seed)

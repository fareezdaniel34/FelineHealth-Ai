"""
Tune U-Net post-processing on the VALIDATION set (no retraining):
  1. threshold   : pixel is "lesion" if probability > threshold
  2. min_blob    : remove connected lesion blobs smaller than this many pixels
                   (removes tiny false specks; if nothing is left -> "no lesion")

Connected-component labelling is implemented here by hand (flood fill).

Usage:
    python ml/src/tune_unet_postprocess.py --model ml/models/unet_v1.npz --base 8
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
import unet_np as un  # noqa: E402


def connected_components(binary):
    """Label 4-connected blobs in a 2-D boolean array. Returns (labels, sizes)."""
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)
    sizes = [0]
    current = 0
    for y in range(H):
        for x in range(W):
            if binary[y, x] and labels[y, x] == 0:
                current += 1
                size = 0
                stack = [(y, x)]
                labels[y, x] = current
                while stack:
                    cy, cx = stack.pop()
                    size += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
                sizes.append(size)
    return labels, np.array(sizes)


def postprocess(prob, threshold, min_blob):
    """prob: (H, W) probabilities -> cleaned boolean mask."""
    binary = prob > threshold
    if min_blob <= 0 or not binary.any():
        return binary
    labels, sizes = connected_components(binary)
    keep = sizes >= min_blob
    keep[0] = False                      # label 0 = background
    return keep[labels]


def score(pred, true):
    """Dice for one image; healthy image counts 1 if nothing predicted, else 0."""
    if true.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return 2 * (pred & true).sum() / (pred.sum() + true.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ml/models/unet_v1.npz")
    p.add_argument("--base", type=int, default=8)
    p.add_argument("--data", default="ml/data/unet_96")
    args = p.parse_args()

    X = np.load(Path(args.data) / "X_val.npy")
    M = np.load(Path(args.data) / "M_val.npy")[..., 0].astype(bool)
    model = un.UNet(base=args.base)
    model.load(args.model)
    prob = model.predict_mask_proba(X.astype(np.float32) / 255.0)
    healthy = M.reshape(len(M), -1).sum(1) == 0
    n_pix = M.shape[1] * M.shape[2]

    rows, best = [], None
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        for frac in (0, 0.001, 0.0025, 0.005, 0.01, 0.02):
            min_blob = int(round(frac * n_pix))
            preds = [postprocess(pr, thr, min_blob) for pr in prob]
            s = np.array([score(pd, t) for pd, t in zip(preds, M)])
            row = dict(threshold=thr, min_blob_px=min_blob, min_blob_pct=frac * 100,
                       dice_all=s.mean(), dice_lesion=s[~healthy].mean(),
                       healthy_clean=s[healthy].mean() if healthy.any() else 1.0)
            rows.append(row)
            if best is None or row["dice_all"] > best["dice_all"] + 1e-9:
                best = row
            print(f"thr {thr:.1f}  min_blob {min_blob:4d}px ({frac:.2%})  "
                  f"Dice all {row['dice_all']:.3f} | lesion imgs {row['dice_lesion']:.3f} | "
                  f"healthy clean {row['healthy_clean']:.0%}")

    Path("ml/results").mkdir(parents=True, exist_ok=True)
    with open("ml/results/unet_postprocess_tuning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    base_row = next(r for r in rows if r["threshold"] == 0.5 and r["min_blob_px"] == 0)
    json.dump({"threshold": best["threshold"], "min_blob_px": best["min_blob_px"],
               "img_size": int(M.shape[1])},
              open("ml/models/unet_postprocess.json", "w"), indent=2)
    print(f"\nBefore (thr 0.5, no cleaning): Dice all {base_row['dice_all']:.3f} | "
          f"lesion {base_row['dice_lesion']:.3f} | healthy clean {base_row['healthy_clean']:.0%}")
    print(f"BEST: threshold {best['threshold']}, min_blob {best['min_blob_px']} px -> "
          f"Dice all {best['dice_all']:.3f} | lesion {best['dice_lesion']:.3f} | "
          f"healthy clean {best['healthy_clean']:.0%}")
    print("Saved: ml/models/unet_postprocess.json")


if __name__ == "__main__":
    main()
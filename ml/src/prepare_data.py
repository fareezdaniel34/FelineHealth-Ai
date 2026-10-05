import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Optional argument: image size, e.g.  python ml/src/prepare_data.py 160
IMG_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 128
CLEAN_DIR = Path("ml/data/clean")
# 128 keeps the old folder name; other sizes get their own folder
OUT_DIR = Path("ml/data/processed") if IMG_SIZE == 128 else Path(f"ml/data/processed_{IMG_SIZE}")
RESULTS_DIR = Path("ml/results")
SPLITS = ["train", "val", "test"]


def load_image(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def main():
    class_names = sorted(p.name for p in (CLEAN_DIR / "train").iterdir() if p.is_dir())
    print("Classes:", class_names)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = {}
    for split in SPLITS:
        X, y = [], []
        for label, cls in enumerate(class_names):
            files = sorted(
                f for f in (CLEAN_DIR / split / cls).iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            for f in files:
                try:
                    X.append(load_image(f))
                    y.append(label)
                except Exception as e:  # corrupted file: skip, but report it
                    print(f"  skipped {f}: {e}")
        X, y = np.stack(X), np.array(y, dtype=np.int64)
        np.save(OUT_DIR / f"X_{split}.npy", X)
        np.save(OUT_DIR / f"y_{split}.npy", y)
        data[split] = (X, y)
        counts = {class_names[i]: int((y == i).sum()) for i in range(len(class_names))}
        print(f"{split:<5} X={X.shape}  {counts}")

    # Class weights (for Keras model.fit(class_weight=...)) to handle imbalance
    y_train = data["train"][1]
    n, k = len(y_train), len(class_names)
    weights = {i: round(n / (k * int((y_train == i).sum())), 4) for i in range(k)}
    with open(OUT_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    with open(OUT_DIR / "class_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    print("Class weights:", {class_names[i]: w for i, w in weights.items()})

    # Visual check: 6 random training images per class
    import matplotlib.pyplot as plt
    X_train = data["train"][0]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(k, 6, figsize=(12, 2.2 * k))
    for r in range(k):
        idx = rng.choice(np.where(y_train == r)[0], 6, replace=False)
        for c, i in enumerate(idx):
            axes[r, c].imshow(X_train[i])
            axes[r, c].axis("off")
        axes[r, 0].set_title(class_names[r], loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"sample_images_{IMG_SIZE}.png", dpi=100)
    print(f"Saved to {OUT_DIR} | check image: {RESULTS_DIR / f'sample_images_{IMG_SIZE}.png'}")


if __name__ == "__main__":
    main()
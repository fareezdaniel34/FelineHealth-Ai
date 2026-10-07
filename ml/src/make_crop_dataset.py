"""
Build a "lesion crop" version of the CNN dataset using the trained U-Net:
for every image in ml/data/clean/{train,val,test}/<class>/, the U-Net finds the
lesion, the photo is cropped around it (whole photo if nothing is found), and
the crop is resized for the CNN.

Same image order and labels as prepare_data.py, so it can be used directly with
train_np_cnn.py:
    python ml/src/make_crop_dataset.py --unet ml/models/unet_v1.npz
    python ml/src/train_np_cnn.py --run_name np_crop_96 --data ml/data/processed_crop_96

Note: the U-Net was trained on 150 of the CNN training images, so crops of the
training set may be slightly better than crops of val/test (mention as a limitation).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from pipeline import centre_crop_resize, lesion_box, load_unet, letterbox, mask_to_original  # noqa: E402
from tune_unet_postprocess import postprocess  # noqa: E402

CLEAN = Path("ml/data/clean")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet", default="ml/models/unet_v1.npz")
    p.add_argument("--postprocess", default="ml/models/unet_postprocess.json")
    p.add_argument("--size", type=int, default=96)
    args = p.parse_args()

    unet, unet_size, pp = load_unet(args.unet, args.postprocess)
    out_dir = Path(f"ml/data/processed_crop_{args.size}")
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = sorted(d.name for d in (CLEAN / "train").iterdir() if d.is_dir())

    for split in ("train", "val", "test"):
        X, y, found = [], [], []
        for label, cls in enumerate(class_names):
            files = sorted(f for f in (CLEAN / split / cls).iterdir()
                           if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            for f in files:
                img = Image.open(f).convert("RGB")
                small, geom = letterbox(img, unet_size)
                prob = unet.predict_mask_proba(np.asarray(small, dtype=np.float32)[None] / 255.0)[0]
                mask = mask_to_original(postprocess(prob, pp["threshold"], pp["min_blob_px"]),
                                        geom, img.size)
                if mask.any():
                    img = img.crop(lesion_box(mask))
                X.append(np.asarray(centre_crop_resize(img, args.size), dtype=np.uint8))
                y.append(label)
                found.append(bool(mask.any()))
        X, y, found = np.stack(X), np.array(y, dtype=np.int64), np.array(found)
        np.save(out_dir / f"X_{split}.npy", X)
        np.save(out_dir / f"y_{split}.npy", y)
        rate = {class_names[i]: f"{found[y == i].mean():.0%}" for i in range(len(class_names))}
        print(f"{split:<5} X={X.shape}  lesion found: {rate}", flush=True)

    y_train = np.load(out_dir / "y_train.npy")
    n, k = len(y_train), len(class_names)
    json.dump(class_names, open(out_dir / "class_names.json", "w"), indent=2)
    json.dump({i: round(n / (k * int((y_train == i).sum())), 4) for i in range(k)},
              open(out_dir / "class_weights.json", "w"), indent=2)

    # preview: 6 crops per class from the training set
    Xtr = np.load(out_dir / "X_train.npy")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(k):
        idx = rng.choice(np.where(y_train == i)[0], 6, replace=False)
        rows.append(np.concatenate(list(Xtr[idx]), axis=1))
    Path("ml/results").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save("ml/results/crop_samples.png")
    print(f"Saved to {out_dir} | preview: ml/results/crop_samples.png "
          f"(rows: {', '.join(class_names)})")


if __name__ == "__main__":
    main()
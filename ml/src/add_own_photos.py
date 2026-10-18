"""
add_own_photos.py
Builds the v2 training data: the ORIGINAL dataset + your own healthy photos
(Random folder only) added to the Health class of the TRAINING split.
val and test are copied unchanged, so v1 and v2 are compared on exactly the
same images. Oki / Sheba / Fifi are NOT used (they stay the real-world test).

Creates:
    ml/data/processed_96_own/        for the full-photo CNN
    ml/data/processed_crop_96_own/   for the crop CNN (crops made by unet_v1)
    ml/results/own_added.csv         which photos were added / dropped
    ml/results/own_added_preview.png full photo + crop of every added photo

Usage (from the project root):
    python ml/src/add_own_photos.py
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.append(str(Path(__file__).parent))
from pipeline import centre_crop_resize, lesion_box, letterbox, load_unet, mask_to_original  # noqa: E402
from tune_unet_postprocess import postprocess                                               # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
HEALTH = "Health"


def open_photo(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)           # fix phone rotation, same as the app
    return img.convert("RGB")


def ahash(img, n=16):
    """Average hash: 256 bits describing the rough light/dark layout of the photo."""
    g = np.asarray(img.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float32)
    return (g > g.mean()).ravel()


def diff(h1, h2):
    return int(np.count_nonzero(h1 != h2))


def processed_dir(prefix, size):
    if prefix == "processed" and size == 128:     # prepare_data.py keeps the old name for 128
        return Path("ml/data/processed")
    return Path(f"ml/data/{prefix}_{size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own_dir", default="ml/data/own_healthy")
    ap.add_argument("--train_folders", default="Random", help="comma-separated subfolders used for TRAINING")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--unet", default="ml/models/unet_v1.npz")
    ap.add_argument("--postprocess", default="ml/models/unet_postprocess.json")
    ap.add_argument("--max_diff", type=int, default=16,
                    help="near-duplicate if hashes differ in <= this many of 256 bits")
    args = ap.parse_args()

    own = Path(args.own_dir)
    train_folders = [s.strip() for s in args.train_folders.split(",")]
    held_out = sorted(d.name for d in own.iterdir() if d.is_dir() and d.name not in train_folders)
    base_full = processed_dir("processed", args.size)
    base_crop = processed_dir("processed_crop", args.size)
    for d in (base_full, base_crop):
        if not (d / "X_train.npy").exists():
            sys.exit(f"Missing {d}/X_train.npy. Run prepare_data.py {args.size} / make_crop_dataset.py first.")
    class_names = json.loads((base_full / "class_names.json").read_text())
    assert json.loads((base_crop / "class_names.json").read_text()) == class_names
    h_label = class_names.index(HEALTH)

    # ---- 1. load training photos + remove near-duplicates (burst shots)
    cand = sorted(f for t in train_folders for f in (own / t).rglob("*") if f.suffix.lower() in EXTS)
    if not cand:
        sys.exit(f"No photos found in {[str(own / t) for t in train_folders]}")
    photos, hashes, log = [], [], []
    for f in cand:
        img = open_photo(f)
        h = ahash(img)
        dup = next((photos[i][0] for i, hk in enumerate(hashes) if diff(h, hk) <= args.max_diff), None)
        if dup is not None:
            log.append({"file": str(f), "status": "dropped: near-duplicate", "of": str(dup)})
            continue
        photos.append((f, img))
        hashes.append(h)

    # ---- 2. safety check: no training photo may look like a held-out (test) photo
    held_files = sorted(f for d in held_out for f in (own / d).rglob("*") if f.suffix.lower() in EXTS)
    held_hashes = [(f, ahash(open_photo(f))) for f in held_files]
    keep = []
    for (f, img), h in zip(photos, hashes):
        clash = next((g for g, hg in held_hashes if diff(h, hg) <= args.max_diff), None)
        if clash is not None:
            log.append({"file": str(f), "status": "dropped: looks like a TEST photo", "of": str(clash)})
            continue
        keep.append((f, img))
        log.append({"file": str(f), "status": "added", "of": ""})
    print(f"Training photos found: {len(cand)} | added: {len(keep)} | "
          f"dropped: {len(cand) - len(keep)}  (held-out test folders: {held_out})")
    for r in log:
        if r["status"] != "added":
            print(f"  {r['status']}: {Path(r['file']).name}  (similar to {Path(r['of']).name})")

    # ---- 3. full-photo arrays and U-Net crop arrays
    unet, unet_size, pp = load_unet(args.unet, args.postprocess)
    X_full, X_crop, n_found = [], [], 0
    for f, img in keep:
        X_full.append(np.asarray(centre_crop_resize(img, args.size), dtype=np.uint8))
        small, geom = letterbox(img, unet_size)
        prob = unet.predict_mask_proba(np.asarray(small, dtype=np.float32)[None] / 255.0)[0]
        mask = mask_to_original(postprocess(prob, pp["threshold"], pp["min_blob_px"]), geom, img.size)
        crop_img = img.crop(lesion_box(mask)) if mask.any() else img
        n_found += bool(mask.any())
        X_crop.append(np.asarray(centre_crop_resize(crop_img, args.size), dtype=np.uint8))
    X_full, X_crop = np.stack(X_full), np.stack(X_crop)
    y_add = np.full(len(keep), h_label, dtype=np.int64)
    print(f"U-Net marked a (false) lesion in {n_found}/{len(keep)} added photos "
          f"-> those crops teach the crop CNN that such areas are healthy")

    # ---- 4. write the new dataset folders (val/test copied unchanged)
    for base, X_add in ((base_full, X_full), (base_crop, X_crop)):
        out = Path(str(base) + "_own")
        out.mkdir(parents=True, exist_ok=True)
        for split in ("val", "test"):
            for kind in ("X", "y"):
                shutil.copyfile(base / f"{kind}_{split}.npy", out / f"{kind}_{split}.npy")
        X_tr = np.concatenate([np.load(base / "X_train.npy"), X_add])
        y_tr = np.concatenate([np.load(base / "y_train.npy"), y_add])
        np.save(out / "X_train.npy", X_tr)
        np.save(out / "y_train.npy", y_tr)
        (out / "class_names.json").write_text(json.dumps(class_names, indent=2))
        n, k = len(y_tr), len(class_names)
        weights = {i: round(n / (k * int((y_tr == i).sum())), 4) for i in range(k)}
        (out / "class_weights.json").write_text(json.dumps(weights, indent=2))
        counts = {c: int((y_tr == i).sum()) for i, c in enumerate(class_names)}
        print(f"{out}: train {X_tr.shape} {counts}")

    # ---- 5. record + preview (top row: full photo, bottom row: crop)
    Path("ml/results").mkdir(parents=True, exist_ok=True)
    with open("ml/results/own_added.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "status", "of"])
        w.writeheader()
        w.writerows(log)
    cols = 12
    tiles = []
    for start in range(0, len(keep), cols):
        a, b = X_full[start:start + cols], X_crop[start:start + cols]
        pad = cols - len(a)
        if pad:
            blank = np.full((pad, args.size, args.size, 3), 255, np.uint8)
            a, b = np.concatenate([a, blank]), np.concatenate([b, blank])
        tiles.append(np.concatenate(list(a), axis=1))
        tiles.append(np.concatenate(list(b), axis=1))
    Image.fromarray(np.concatenate(tiles, axis=0)).save("ml/results/own_added_preview.png")
    print("Saved: ml/results/own_added.csv, ml/results/own_added_preview.png")


if __name__ == "__main__":
    main()
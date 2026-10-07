"""
Turn the Roboflow "COCO Segmentation" export of your lesion masks into U-Net
training data (images + binary masks). Images are letterboxed (padded to a
square, nothing cut off) and resized.

Usage (from the FelineHealth-Ai project root, with venv active):
    python ml/src/prepare_unet_data.py ml/data/masks_export 96

Expects:
    ml/data/masks_export/          <- extracted Roboflow export
    ml/data/unet_selection.csv     <- from the annotation pack
    ml/data/raw/                   <- original dataset (for the Health images)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EXPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ml/data/masks_export")
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 96
SELECTION = Path("ml/data/unet_selection.csv")
RAW = Path("ml/data/raw")
OUT = Path(f"ml/data/unet_{SIZE}")
RESULTS = Path("ml/results")


def letterbox(img, size, resample, fill=0):
    """Keep the WHOLE photo: pad the short side to make a square, then resize.
    (A centre crop would cut off lesions near the edges, e.g. scabies on ears.)
    The app must use the same function before running the U-Net."""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new(img.mode, (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas.resize((size, size), resample)


def unet_id_from_filename(fn):
    # Roboflow renames "tr_flea_001.jpg" to "tr_flea_001_jpg.rf.<hash>.jpg"
    if ".rf." in fn:
        fn = fn.split(".rf.")[0]
        for ext in ("_jpg", "_jpeg", "_png"):
            if fn.endswith(ext):
                return fn[: -len(ext)]
        return fn
    return Path(fn).stem


def main():
    if SIZE % 8:
        sys.exit("SIZE must be divisible by 8 for the U-Net")
    sel = list(csv.DictReader(open(SELECTION, encoding="utf-8")))

    # 1. read every COCO json in the export (Roboflow puts them in train/valid/test)
    images = {}                                    # unet_id -> (image path, polygons)
    for js in EXPORT.rglob("_annotations.coco.json"):
        coco = json.load(open(js, encoding="utf-8"))
        polys = {}
        for a in coco["annotations"]:
            for seg in a.get("segmentation") or []:
                if isinstance(seg, list) and len(seg) >= 6:
                    polys.setdefault(a["image_id"], []).append(seg)
        for im in coco["images"]:
            uid = unet_id_from_filename(im["file_name"])
            images[uid] = (js.parent / im["file_name"], polys.get(im["id"], []))

    data = {s: {"X": [], "M": [], "ids": []} for s in ("train", "val", "test")}
    missing, no_polygon = [], []
    for r in sel:
        uid, split = r["unet_id"], r["unet_split"]
        if r["needs_mask"] == "True":
            if uid not in images:
                missing.append(uid); continue
            path, polys = images[uid]
            if not polys:                     # you found no visible lesion -> skip
                no_polygon.append(uid); continue
            img = Image.open(path).convert("RGB")
            mask = Image.new("L", img.size, 0)
            draw = ImageDraw.Draw(mask)
            for seg in polys:
                draw.polygon([(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)], fill=1)
        else:                                 # Health: empty mask
            img = Image.open(RAW / r["orig_path"]).convert("RGB")
            mask = Image.new("L", img.size, 0)
        x = np.asarray(letterbox(img, SIZE, Image.BILINEAR), dtype=np.uint8)
        m = np.asarray(letterbox(mask, SIZE, Image.NEAREST), dtype=np.uint8)
        data[split]["X"].append(x)
        data[split]["M"].append(m[..., None])
        data[split]["ids"].append(uid)

    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for split, d in data.items():
        X, M = np.stack(d["X"]), np.stack(d["M"])
        np.save(OUT / f"X_{split}.npy", X)
        np.save(OUT / f"M_{split}.npy", M)
        json.dump(d["ids"], open(OUT / f"ids_{split}.json", "w"))
        healthy = sum(i.split("_")[1] == "health" for i in d["ids"])
        cov = M.reshape(len(M), -1).mean(1)
        print(f"{split:<5} X={X.shape}  healthy(empty mask)={healthy}  "
              f"lesion area: mean {cov[cov > 0].mean():.1%} of image")

    if missing:
        print(f"\nWARNING {len(missing)} selected images not found in the export, e.g. {missing[:5]}")
    if no_polygon:
        print(f"\nNOTE {len(no_polygon)} images had no polygon and were skipped: {no_polygon}")

    # preview: 8 training images with the mask in red
    X, M = np.stack(data["train"]["X"]), np.stack(data["train"]["M"])
    rng = np.random.default_rng(0)
    pick = rng.choice(len(X), min(8, len(X)), replace=False)
    tiles = []
    for i in pick:
        ov = X[i].astype(float)
        ov[M[i, ..., 0] > 0] = ov[M[i, ..., 0] > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        tiles.append(np.concatenate([X[i], ov.astype(np.uint8)], axis=0))
    Image.fromarray(np.concatenate(tiles, axis=1)).resize(
        (len(tiles) * 160, 320), Image.NEAREST).save(RESULTS / "unet_masks_preview.png")
    print(f"\nSaved to {OUT} | preview: {RESULTS / 'unet_masks_preview.png'} (top: image, bottom: mask)")


if __name__ == "__main__":
    main()
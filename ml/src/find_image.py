"""
find_image.py
Where did this photo end up after cleaning (train / val / test), and what did the
pipeline predict for it?  Matches by CONTENT, so renamed files are found too.

Usage (from the project root; use quotes because the path has spaces):
    python ml/src/find_image.py "C:/Users/Asus/Desktop/FYP - FelineHealth/Dataset/CAT SKIN DISEASE/valid/Ringworm/xxx.jpg"
    python ml/src/find_image.py "path/one.jpg" "path/two.jpg"
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CLEAN = Path("ml/data/clean")
RESULTS = Path("ml/results")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def ahash(path, n=16):
    g = np.asarray(Image.open(path).convert("L").resize((n, n), Image.BILINEAR), dtype=np.float32)
    return (g > g.mean()).ravel()


def prediction(split, filename):
    f = RESULTS / f"pipeline_{split}_per_image.csv"
    if split not in ("val", "test") or not f.exists():
        return None
    with open(f, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["file"] == filename:
                return row
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    index = [(p, p.relative_to(CLEAN).parts) for p in CLEAN.rglob("*") if p.suffix.lower() in EXTS]
    md5s = {}
    for p, _ in index:
        md5s.setdefault(md5(p), []).append(p)
    hashes = None
    for q in sys.argv[1:]:
        q = Path(q)
        print(f"\n{q.name}")
        if not q.exists():
            print("  file not found")
            continue
        hits = md5s.get(md5(q), [])
        how = "identical file"
        if not hits:                                     # renamed / re-saved: compare by look
            if hashes is None:
                hashes = [(p, ahash(p)) for p, _ in index]
            h = ahash(q)
            d = sorted((int(np.count_nonzero(h != hp)), p) for p, hp in hashes)
            hits = [p for diff, p in d if diff <= 10]
            how = "near-identical image"
        if not hits:
            print("  NOT in ml/data/clean (removed during cleaning, or a different image)")
            continue
        for p in hits:
            split, cls = p.relative_to(CLEAN).parts[:2]
            print(f"  {how}: split = {split.upper()}, class folder = {cls}, file = {p.name}")
            row = prediction(split, p.name)
            if row:
                print(f"  pipeline C predicted {row['pred_combined']} ({float(row['conf_combined']):.0%}), "
                      f"lesion found: {row['lesion_found']}")
            elif split == "train":
                print("  -> TRAINING image: the model learned from it; do not use it as an example")


if __name__ == "__main__":
    main()
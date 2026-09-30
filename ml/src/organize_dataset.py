"""
Build the clean, leakage-free dataset for FelineHealth-AI from split.csv.

Usage (from the FelineHealth-Ai project root, with venv active):
    python ml/src/organize_dataset.py

Expects:
    ml/data/raw/            <- extracted Roboflow "Folder Structure" export
        train/<class>/...      (folders: train, valid, test)
        valid/<class>/...
        test/<class>/...
    ml/data/split.csv       <- the split file

Creates:
    ml/data/clean/train/<class>/...
    ml/data/clean/val/<class>/...
    ml/data/clean/test/<class>/...
"""
import csv
import shutil
from collections import Counter
from pathlib import Path

RAW_DIR = Path("ml/data/raw")
SPLIT_CSV = Path("ml/data/split.csv")
OUT_DIR = Path("ml/data/clean")


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)  # rebuild from scratch every time

    counts = Counter()
    missing = []

    with open(SPLIT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["keep"] != "True":
                continue
            src = RAW_DIR / row["orig_path"]
            dst = OUT_DIR / row["new_split"] / row["class"] / row["filename"]
            if not src.exists():
                missing.append(str(src))
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            counts[(row["new_split"], row["class"])] += 1

    print(f"{'split':<6} {'class':<14} images")
    for (split, cls), n in sorted(counts.items()):
        print(f"{split:<6} {cls:<14} {n}")
    print(f"Total copied: {sum(counts.values())}")

    if missing:
        print(f"\nWARNING: {len(missing)} files not found, e.g.:")
        for m in missing[:5]:
            print("  ", m)


if __name__ == "__main__":
    main()

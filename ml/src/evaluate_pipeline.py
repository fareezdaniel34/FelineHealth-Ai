"""
Key experiment: does U-Net segmentation help the CNN classify?

Compares on a split of the CLEAN CNN dataset (default: val):
    A. full     : CNN on the whole photo (CNN alone)
    B. crop     : CNN on the crop around the U-Net lesion
    C. combined : average of A and B

Usage:
    python ml/src/evaluate_pipeline.py --cnn ml/models/np_v2_96.npz --unet ml/models/unet_v1.npz
    (use --split test ONLY ONCE at the very end, for the final report numbers)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from np_metrics import classification_metrics, format_report   # noqa: E402
from pipeline import FelineHealthPipeline                       # noqa: E402

RESULTS = Path("ml/results")
MODES = ("full", "crop", "combined")


def save_confusion(cm, class_names, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn", default="ml/models/np_v2_96.npz")
    p.add_argument("--unet", default="ml/models/unet_v1.npz")
    p.add_argument("--crop_cnn", default=None,
                   help="CNN trained on lesion crops (if not given, the full-photo CNN is used for B)")
    p.add_argument("--postprocess", default="ml/models/unet_postprocess.json")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--clean_dir", default="ml/data/clean")
    args = p.parse_args()

    pipe = FelineHealthPipeline(args.cnn, args.unet, args.postprocess, args.crop_cnn)
    names = pipe.class_names
    y_true, preds, rows = [], {m: [] for m in MODES}, []
    for label, cls in enumerate(names):
        files = sorted(f for f in (Path(args.clean_dir) / args.split / cls).iterdir()
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for f in files:
            out = pipe.predict(Image.open(f))
            y_true.append(label)
            row = {"file": f.name, "true": cls, "lesion_found": out["lesion_found"],
                   "lesion_area_pct": round(out["lesion_area_pct"], 2)}
            for m in MODES:
                k = int(out[m].argmax())
                preds[m].append(k)
                row[f"pred_{m}"] = names[k]
                row[f"conf_{m}"] = round(float(out[m][k]), 3)
            rows.append(row)
        print(f"{cls}: {len(files)} images done", flush=True)

    y_true = np.array(y_true)
    RESULTS.mkdir(parents=True, exist_ok=True)
    lines = []
    for m in MODES:
        met = classification_metrics(y_true, np.array(preds[m]), len(names))
        title = {"full": "A. CNN on full photo", "crop": "B. CNN on U-Net crop",
                 "combined": "C. Combined (A+B)/2"}[m]
        lines += [f"=== {title} ===", format_report(met, names), ""]
        save_confusion(met["confusion_matrix"], names, f"{title} (acc {met['accuracy']:.1%})",
                       RESULTS / f"pipeline_{args.split}_confusion_{m}.png")
        print(f"{title:<24} accuracy {met['accuracy']:.1%}  macro-F1 {met['macro_f1']:.3f}  "
              f"Scabies recall {met['recall'][names.index('Scabies')]:.2f}")
    found = np.array([r["lesion_found"] for r in rows])
    for label, cls in enumerate(names):
        lines.append(f"U-Net found a lesion in {found[y_true == label].mean():.0%} of {cls} images")
    report = "\n".join(lines)
    (RESULTS / f"pipeline_{args.split}_report.txt").write_text(report)
    with open(RESULTS / f"pipeline_{args.split}_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n" + "\n".join(lines[-len(names):]))
    print(f"\nFull report: {RESULTS / f'pipeline_{args.split}_report.txt'}")


if __name__ == "__main__":
    main()
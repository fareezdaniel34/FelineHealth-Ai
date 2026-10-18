"""
evaluate_ensemble.py
Can we raise accuracy WITHOUT retraining? Compares 4 versions of strategy C
(full-photo CNN + crop CNN, averaged) on the VALIDATION set:

    single      : seed 42 models only (the current app)
    single_tta  : seed 42 + test-time augmentation (photo + mirror image)
    ens3        : average of the 3 seed models (42, s1, s2)
    ens3_tta    : 3 seeds + mirror image

All 4 use the same U-Net mask and crop box. Reports raw accuracy / macro-F1 and
the app's real decision rules (min_confidence 0.45 + empty-mask gate).
Also writes one per-image CSV per version (same format as evaluate_pipeline.py,
so gate_rule_analysis.py can read it).

DECISION RULE (fixed before running): use the version with the highest
validation macro-F1; if it is within 1 point of a simpler version, use the
simpler one (order of simplicity: single < single_tta < ens3 < ens3_tta).

Usage (from the project root):
    python ml/src/evaluate_ensemble.py
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from np_metrics import classification_metrics                      # noqa: E402
from pipeline import FelineHealthPipeline, centre_crop_resize, lesion_box, load_cnn  # noqa: E402

MODELS = Path("ml/models")
FULL = ["np_v2_96.npz", "np_v2_96_s1.npz", "np_v2_96_s2.npz"]        # first = seed 42 (app)
CROP = ["np_crop_96.npz", "np_crop_96_s1.npz", "np_crop_96_s2.npz"]
VERSIONS = ["single", "single_tta", "ens3", "ens3_tta"]
MIN_CONF = 0.45
HEALTH = "Health"


def probs_orig_and_mirror(model, size, img):
    """Returns (2, n_classes): row 0 = photo, row 1 = mirror image."""
    x = np.asarray(centre_crop_resize(img, size), dtype=np.float32)[None] / 255.0
    x2 = np.ascontiguousarray(np.concatenate([x, x[:, :, ::-1, :]], axis=0))
    return model.predict_proba(x2)


def combine(full, crop, version):
    """full, crop: arrays (n_models, 2, n_classes). Returns strategy C probs."""
    models = slice(0, 1) if version.startswith("single") else slice(None)
    views = slice(None) if version.endswith("tta") else slice(0, 1)
    p_full = full[models, views].mean(axis=(0, 1))
    p_crop = crop[models, views].mean(axis=(0, 1))
    return (p_full + p_crop) / 2


def app_decision(label, conf, lesion_found):
    if conf < MIN_CONF:
        return "uncertain"
    if label == HEALTH:
        return "healthy"
    if not lesion_found:                       # empty-mask gate ("always")
        return "uncertain"
    return "disease"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--clean_dir", default="ml/data/clean")
    ap.add_argument("--unet", default="ml/models/unet_v1.npz")
    ap.add_argument("--postprocess", default="ml/models/unet_postprocess.json")
    args = ap.parse_args()
    if args.split == "test":
        print("WARNING: test split. Run this ONCE, only after choosing the version on val.\n")

    for f in FULL + CROP:
        if not (MODELS / f).exists():
            sys.exit(f"Missing model file: {MODELS / f}")

    # U-Net + helpers from the normal pipeline (seed 42 models, as in the app)
    pipe = FelineHealthPipeline(str(MODELS / FULL[0]), args.unet, args.postprocess,
                                str(MODELS / CROP[0]))
    names = pipe.class_names
    full_models = [load_cnn(str(MODELS / f)) for f in FULL]
    crop_models = [load_cnn(str(MODELS / f)) for f in CROP]
    for _, _, cn in full_models + crop_models:
        assert cn == names, f"Class order differs between models: {cn} vs {names}"

    y_true, files, found_list = [], [], []
    probs = {v: [] for v in VERSIONS}
    t0 = time.time()
    for label, cls in enumerate(names):
        img_files = sorted(f for f in (Path(args.clean_dir) / args.split / cls).iterdir()
                           if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for f in img_files:
            img = Image.open(f).convert("RGB")
            mask = pipe.segment(img)
            found = bool(mask.any())
            crop_img = img.crop(lesion_box(mask)) if found else img
            full = np.stack([probs_orig_and_mirror(m, s, img) for m, s, _ in full_models])
            crop = np.stack([probs_orig_and_mirror(m, s, crop_img) for m, s, _ in crop_models])
            for v in VERSIONS:
                probs[v].append(combine(full, crop, v))
            y_true.append(label)
            files.append(f.name)
            found_list.append(found)
        print(f"{cls}: {len(img_files)} images done", flush=True)
    print(f"({time.time() - t0:.0f} s)\n")

    y_true = np.array(y_true)
    found_arr = np.array(found_list)
    out_dir = Path("ml/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    h = names.index(HEALTH)

    header = (f"{'version':<11} {'acc':>6} {'macroF1':>8} {'ScabRec':>8} {'meanConf':>9} "
              f"{'unclear':>8} {'confAcc':>8} {'healthFA':>9}")
    print(header)
    print("-" * len(header))
    summary = []
    for v in VERSIONS:
        P = np.array(probs[v])
        pred = P.argmax(1)
        conf = P.max(1)
        met = classification_metrics(y_true, pred, len(names))
        dec = np.array([app_decision(names[k], c, fd) for k, c, fd in zip(pred, conf, found_arr)])
        confident = dec != "uncertain"
        conf_acc = (pred[confident] == y_true[confident]).mean() if confident.any() else 0.0
        health_fa = int(((y_true == h) & (dec == "disease")).sum())
        n_health = int((y_true == h).sum())
        row = {"version": v, "accuracy": met["accuracy"], "macro_f1": met["macro_f1"],
               "scabies_recall": met["recall"][names.index("Scabies")],
               "mean_conf": conf.mean(), "unclear_pct": 100 * (~confident).mean(),
               "confident_acc": conf_acc, "health_false_alarm": health_fa}
        summary.append(row)
        print(f"{v:<11} {met['accuracy']:6.1%} {met['macro_f1']:8.3f} "
              f"{row['scabies_recall']:8.2f} {conf.mean():9.2f} {row['unclear_pct']:7.1f}% "
              f"{conf_acc:8.1%} {health_fa:>4}/{n_health:<4}")

        with open(out_dir / f"ensemble_{args.split}_{v}_per_image.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "true", "lesion_found", "pred_combined", "conf_combined"]
                       + [f"p_{c}" for c in names])
            for i in range(len(files)):
                w.writerow([files[i], names[y_true[i]], found_arr[i], names[pred[i]],
                            round(float(conf[i]), 4)] + [round(float(p), 4) for p in P[i]])

    with open(out_dir / f"ensemble_{args.split}_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # pre-specified choice
    best = max(summary, key=lambda r: r["macro_f1"])
    for r in summary:                          # simplest version within 1 F1 point of the best
        if best["macro_f1"] - r["macro_f1"] <= 0.01:
            chosen = r
            break
    print(f"\nHighest macro-F1: {best['version']} ({best['macro_f1']:.3f})")
    print(f"Chosen by the rule (simplest within 0.01 F1): {chosen['version']}")
    print(f"Per-image files: {out_dir}/ensemble_{args.split}_<version>_per_image.csv")


if __name__ == "__main__":
    main()
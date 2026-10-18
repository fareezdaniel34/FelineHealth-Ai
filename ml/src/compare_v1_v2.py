"""
compare_v1_v2.py
v1 = original training data, v2 = + 44 own healthy photos (Random).

PART 1 (decides): strategy C on the VALIDATION set, 3 seed pairs per version.
    Rule fixed before training: adopt v2 if mean macro-F1(v2) >= mean macro-F1(v1) - 0.01
PART 2 (reported only, does NOT change the decision): the real app (seed 42 models,
    threshold 0.45 + empty-mask gate) on the held-out cats Fifi, Oki, Sheba.

Usage (from the project root):
    python ml/src/compare_v1_v2.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(ROOT / "backend"))
from np_metrics import classification_metrics                          # noqa: E402
from pipeline import FelineHealthPipeline, centre_crop_resize, lesion_box, load_cnn  # noqa: E402

MODELS = Path("ml/models")
VERSIONS = {
    "v1": [("np_v2_96", "np_crop_96"), ("np_v2_96_s1", "np_crop_96_s1"), ("np_v2_96_s2", "np_crop_96_s2")],
    "v2": [("np_v2_96_own", "np_crop_96_own"), ("np_v2_96_own_s1", "np_crop_96_own_s1"),
           ("np_v2_96_own_s2", "np_crop_96_own_s2")],
}
SEEDS = ["42", "1", "2"]
MAX_DROP = 0.01
HELD_OUT = ["Fifi", "Oki", "Sheba"]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def cnn_probs(model, size, img):
    x = np.asarray(centre_crop_resize(img, size), dtype=np.float32)[None] / 255.0
    return model.predict_proba(x)[0]


def part1_validation():
    for pairs in VERSIONS.values():
        for full, crop in pairs:
            for name in (full, crop):
                if not (MODELS / f"{name}.npz").exists():
                    sys.exit(f"Missing model: {MODELS / name}.npz")
    seg = FelineHealthPipeline(str(MODELS / "np_v2_96.npz"), "ml/models/unet_v1.npz",
                               "ml/models/unet_postprocess.json", str(MODELS / "np_crop_96.npz"))
    names = seg.class_names
    models = {v: [(load_cnn(str(MODELS / f"{f}.npz")), load_cnn(str(MODELS / f"{c}.npz")))
                  for f, c in pairs] for v, pairs in VERSIONS.items()}
    for v in models:
        for (fm, cm) in models[v]:
            assert fm[2] == names and cm[2] == names, "class order differs"

    # U-Net once per image (same for all models), then every CNN pair
    y_true, preds = [], {v: [[] for _ in SEEDS] for v in VERSIONS}
    for label, cls in enumerate(names):
        files = sorted(f for f in (Path("ml/data/clean/val") / cls).iterdir()
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for f in files:
            img = Image.open(f).convert("RGB")
            mask = seg.segment(img)
            crop_img = img.crop(lesion_box(mask)) if mask.any() else img
            for v in VERSIONS:
                for s, ((fm, fs, _), (cm, cs, _)) in enumerate(models[v]):
                    p = (cnn_probs(fm, fs, img) + cnn_probs(cm, cs, crop_img)) / 2
                    preds[v][s].append(int(p.argmax()))
            y_true.append(label)
    y_true = np.array(y_true)

    print("PART 1  Validation, strategy C (full + crop CNN), per seed")
    print(f"{'':4} " + "  ".join(f"seed {s:>2}: acc / F1   " for s in SEEDS) + "  mean acc / F1")
    means = {}
    for v in VERSIONS:
        accs, f1s, cells = [], [], []
        for s in range(len(SEEDS)):
            m = classification_metrics(y_true, np.array(preds[v][s]), len(names))
            accs.append(m["accuracy"]); f1s.append(m["macro_f1"])
            cells.append(f"{m['accuracy']:6.1%} / {m['macro_f1']:.3f}")
        means[v] = (np.mean(accs), np.mean(f1s), np.std(f1s))
        print(f"{v:4} " + "    ".join(cells) + f"    {means[v][0]:6.1%} / {means[v][1]:.3f} (sd {means[v][2]:.3f})")
    diff = means["v2"][1] - means["v1"][1]
    adopt = diff >= -MAX_DROP
    print(f"\nmacro-F1 change v2 - v1: {diff:+.3f}  (rule: must be >= -{MAX_DROP})")
    print("DECISION:", "ADOPT v2" if adopt else "KEEP v1")
    return adopt


def part2_real_world():
    from inference import FelineHealthService
    files = sorted(f for d in HELD_OUT for f in (Path("ml/data/own_healthy") / d).rglob("*")
                   if f.suffix.lower() in EXTS)
    print(f"\nPART 2  Real-world held-out cats {HELD_OUT}: {len(files)} photos "
          f"(app rules, seed 42 models; reported only)")
    print(f"{'':4} {'healthy':>8} {'unclear':>8} {'disease':>8} {'top=Health':>11}")
    for v, pairs in VERSIONS.items():
        full, crop = pairs[0]
        svc = FelineHealthService({"cnn": MODELS / f"{full}.npz", "crop_cnn": MODELS / f"{crop}.npz"})
        res = [svc.analyse(f.read_bytes()) for f in files]
        c = Counter(r["result_type"] for r in res)
        top_h = sum(r["prediction"]["label"] == "Health" for r in res)
        print(f"{v:4} {c['healthy']:>8} {c['uncertain']:>8} {c['disease']:>8} {top_h:>11}")


if __name__ == "__main__":
    part1_validation()
    part2_real_world()
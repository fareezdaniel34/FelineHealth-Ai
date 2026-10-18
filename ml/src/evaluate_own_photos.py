"""
evaluate_own_photos.py
Real-world check: runs the EXACT app pipeline (backend/inference.py, same
thresholds and empty-mask gate) on your own photos and reports how often the
app says healthy / unclear / disease.

Folder layout (one subfolder per cat, so photos of the same cat stay together):
    ml/data/own_healthy/
        oyen/    IMG_001.jpg, IMG_002.jpg, ...
        comot/   ...
Usage (from the project root):
    python ml/src/evaluate_own_photos.py
    python ml/src/evaluate_own_photos.py --folder ml/data/own_healthy --true Health
"""
import argparse
import base64
import csv
import io
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]               # FelineHealth-Ai/
sys.path.append(str(ROOT / "backend"))
from inference import FelineHealthService                # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion k/n (Wilson score)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k:>3}/{n:<3} = {100 * k / max(n, 1):5.1f}%  (95% CI {100 * lo:4.1f}-{100 * hi:4.1f}%)"


def data_url_to_image(url):
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


def contact_sheet(items, path, thumb=180, cols=6):
    """Grid of the photos the app did NOT call healthy, with the app's answer."""
    if not items:
        return False
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 34)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (img, line1, line2) in enumerate(items):
        img = img.copy()
        img.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 34)
        sheet.paste(img, (x + (thumb - img.width) // 2, y + (thumb - img.height) // 2))
        draw.text((x + 4, y + thumb + 2), line1[:30], fill=(0, 0, 0))
        draw.text((x + 4, y + thumb + 17), line2[:30], fill=(90, 90, 90))
    sheet.save(path, quality=88)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="ml/data/own_healthy")
    ap.add_argument("--true", default="Health", help="true class of ALL photos in the folder")
    ap.add_argument("--out", default="ml/results")
    args = ap.parse_args()

    folder, out_dir = Path(args.folder), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in EXTS)
    skipped = sorted(f.name for f in folder.rglob("*")
                     if f.is_file() and f.suffix.lower() not in EXTS)
    if not files:
        sys.exit(f"No JPG/PNG/WEBP photos found in {folder}")

    svc = FelineHealthService()
    print(f"Settings used: min_confidence={svc.cfg['min_confidence']}, "
          f"empty_mask_gate={svc.cfg.get('empty_mask_gate', False)}")
    print(f"Photos: {len(files)}" + (f"  (skipped non-image files: {skipped})" if skipped else ""))

    rows, sheet_items = [], []
    for i, f in enumerate(files, 1):
        cat = f.parent.name if f.parent != folder else "(no folder)"
        try:
            r = svc.analyse(f.read_bytes())
        except Exception as e:                                # unreadable photo etc.
            print(f"  ERROR {f.name}: {e}")
            rows.append({"file": f.name, "cat": cat, "result_type": "error", "error": str(e)})
            continue
        pred = r["prediction"]
        row = {"file": f.name, "cat": cat, "result_type": r["result_type"],
               "uncertain_reason": r.get("uncertain_reason") or "",
               "label": pred["label"], "confidence": pred["confidence"],
               "lesion_found": r["lesion"]["found"], "lesion_area_pct": r["lesion"]["area_pct"]}
        row.update({f"p_{c}": p for c, p in pred["probabilities"].items()})
        rows.append(row)
        if r["result_type"] != "healthy":
            shown = r["images"]["overlay"] or r["images"]["original"]
            sheet_items.append((data_url_to_image(shown),
                                f"{r['result_type']}: {pred['label']} {pred['confidence']:.0%}",
                                f"{cat}/{f.name}"))
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)} done", flush=True)

    # ---- save per-photo results
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    csv_path = out_dir / "own_photos_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- summary
    ok = [r for r in rows if r["result_type"] != "error"]
    n = len(ok)
    types = Counter(r["result_type"] for r in ok)
    reasons = Counter(r["uncertain_reason"] for r in ok if r["result_type"] == "uncertain")
    wrong_labels = Counter(r["label"] for r in ok if r["result_type"] == "disease")
    top_correct = sum(r["label"] == args.true for r in ok)
    lesion_found = sum(bool(r["lesion_found"]) for r in ok)

    lines = [f"Own photos, true class = {args.true}   (photos analysed: {n}, errors: {len(rows) - n})",
             f"Settings: min_confidence={svc.cfg['min_confidence']}, "
             f"empty_mask_gate={svc.cfg.get('empty_mask_gate', False)}", ""]
    if args.true == "Health":
        lines += ["App result",
                  f"  healthy   (correct)        {pct(types['healthy'], n)}",
                  f"  unclear                    {pct(types['uncertain'], n)}",
                  f"     low confidence          {reasons['low_confidence']}",
                  f"     no lesion found (gate)  {reasons['no_lesion_found']}",
                  f"  disease   (FALSE ALARM)    {pct(types['disease'], n)}"]
        for lab, k in wrong_labels.most_common():
            lines.append(f"     as {lab:<22}{k}")
        lines += ["",
                  f"Top class is Health (ignoring thresholds)  {pct(top_correct, n)}",
                  f"U-Net marked a lesion                      {pct(lesion_found, n)}"]
    else:
        lines += [f"Confident and correct   {pct(sum(r['result_type'] == 'disease' and r['label'] == args.true for r in ok), n)}",
                  f"Top class correct       {pct(top_correct, n)}"]

    per_cat = defaultdict(Counter)
    for r in ok:
        per_cat[r["cat"]][r["result_type"]] += 1
    lines += ["", "Per cat (healthy / unclear / disease)"]
    for cat, c in sorted(per_cat.items()):
        lines.append(f"  {cat:<20} {c['healthy']:>3} / {c['uncertain']:>3} / {c['disease']:>3}")

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "own_photos_report.txt").write_text(report, encoding="utf-8")
    print(f"\nPer-photo results: {csv_path}")
    if contact_sheet(sheet_items, out_dir / "own_photos_not_healthy.jpg"):
        print(f"Photos NOT called healthy: {out_dir / 'own_photos_not_healthy.jpg'}")


if __name__ == "__main__":
    main()
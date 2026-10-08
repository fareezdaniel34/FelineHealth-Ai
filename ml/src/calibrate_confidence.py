"""
Choose the 'uncertain' confidence limit for the app, using the validation set.
For each threshold: how many photos get a confident answer, and how accurate
those confident answers are.

Usage (after evaluate_pipeline.py on the val split):
    python ml/src/calibrate_confidence.py
"""
import csv

rows = list(csv.DictReader(open("ml/results/pipeline_val_per_image.csv", encoding="utf-8")))
n = len(rows)
print(f"{'threshold':>9} {'confident':>10} {'acc(confident)':>15} {'uncertain':>10} {'acc(uncertain)':>15}")
for thr in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
    conf = [r for r in rows if float(r["conf_combined"]) >= thr]
    unc = [r for r in rows if float(r["conf_combined"]) < thr]
    acc = lambda rs: sum(r["pred_combined"] == r["true"] for r in rs) / len(rs) if rs else float("nan")
    print(f"{thr:>9.2f} {len(conf) / n:>10.0%} {acc(conf):>15.1%} {len(unc) / n:>10.0%} {acc(unc):>15.1%}")
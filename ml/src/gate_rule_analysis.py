"""
gate_rule_analysis.py
Checks the rule: "if the U-Net finds no lesion and the top class is a disease
with probability below GATE, return 'uncertain' instead of a disease".

Reads the per-image file written by evaluate_pipeline.py (validation split,
run with the app models: np_v2_96 + np_crop_96 + unet_v1).
Uses strategy C (combined), the same as the app.

Usage (from the project root):
    python ml/src/gate_rule_analysis.py
    python ml/src/gate_rule_analysis.py ml/results/pipeline_val_per_image.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HEALTH = "Health"
MIN_CONF = 0.45                      # current app threshold
GATES = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]  # 0.45 = no gate, 1.01 = always
MAX_LOST = 2                         # decision rule, fixed BEFORE looking at results


def decide(top, conf, empty, gate):
    """Returns 'uncertain' or the predicted label for every image."""
    out = np.where(conf < MIN_CONF, "uncertain", top).astype(object)
    gated = (out != "uncertain") & (top != HEALTH) & empty & (conf < gate)
    out[gated] = "uncertain"
    return out


def summarise(true, out):
    is_h = true == HEALTH
    confident = out != "uncertain"
    correct = (out == true) & confident
    return {
        "uncertain": int((~confident).sum()),
        "uncertain_%": 100 * (~confident).mean(),
        "confident_acc_%": 100 * correct.sum() / max(confident.sum(), 1),
        "disease_correct": int((correct & ~is_h).sum()),
        # healthy cat confidently called a disease
        "health_false_alarm": int((is_h & confident & (out != HEALTH)).sum()),
        # any confident disease result that is wrong
        "wrong_disease_result": int((confident & (out != HEALTH) & (out != true)).sum()),
    }


def main(path):
    df = pd.read_csv(path)
    true = df["true"].astype(str).values
    top = df["pred_combined"].astype(str).values
    conf = df["conf_combined"].astype(float).values
    empty = ~df["lesion_found"].astype(str).str.lower().eq("true").values

    print(f"Validation images: {len(df)}\n")

    # 1) How often is the U-Net mask empty, per TRUE class?
    print("U-Net empty-mask rate per true class")
    for c in sorted(set(true)):
        m = true == c
        print(f"  {c:<14} {empty[m].sum():>3}/{m.sum():<3} empty ({100*empty[m].mean():5.1f}%)")
    print("  (Health: high is good. Diseases: high means the gate would hide real cases.)\n")

    # 2) Sweep the gate
    base = summarise(true, decide(top, conf, empty, MIN_CONF))
    rows = []
    for g in GATES:
        s = summarise(true, decide(top, conf, empty, g))
        s["gate"] = "always" if g > 1 else f"{g:.2f}"
        s["lost_disease_correct"] = base["disease_correct"] - s["disease_correct"]
        s["removed_wrong"] = base["wrong_disease_result"] - s["wrong_disease_result"]
        rows.append(s)
    res = pd.DataFrame(rows)[["gate", "uncertain", "uncertain_%", "confident_acc_%",
                              "disease_correct", "lost_disease_correct",
                              "health_false_alarm", "wrong_disease_result", "removed_wrong"]]
    print("Gate sweep (0.45 = current app, no gate)")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # 3) Pre-specified choice: largest gate losing at most MAX_LOST correct disease results
    best = res[res["lost_disease_correct"] <= MAX_LOST].iloc[-1]
    print(f"\nDecision rule (fixed in advance): largest gate with lost_disease_correct <= {MAX_LOST}")
    if best["gate"] == "0.45" or best["removed_wrong"] == 0:
        print("-> Keep the current app. The gate does not help on validation.")
    else:
        print(f"-> Use gate {best['gate']}: removes {int(best['removed_wrong'])} wrong disease result(s), "
              f"loses {int(best['lost_disease_correct'])} correct disease result(s).")

    out_path = Path(path).with_name(Path(path).stem + "_gate_sweep.csv")
    res.to_csv(out_path, index=False)
    print(f"\nSaved table: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml/results/pipeline_val_per_image.csv")
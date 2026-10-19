"""
Evaluation metrics written from scratch (NumPy only).
"""
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    """cm[i, j] = number of images of true class i predicted as class j."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_metrics(y_true, y_pred, num_classes):
    """
    Per class:  precision = TP / (TP + FP)
                recall    = TP / (TP + FN)
                F1        = 2 * precision * recall / (precision + recall)
    Overall:    accuracy  = correct / total,  macro = average over classes
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "confusion_matrix": cm,
        "accuracy": float(tp.sum() / cm.sum()),
        "precision": precision, "recall": recall, "f1": f1,
        "support": cm.sum(axis=1),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def format_report(m, class_names):
    lines = [f"{'':<14}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}", ""]
    for i, c in enumerate(class_names):
        lines.append(f"{c:<14}{m['precision'][i]:>10.3f}{m['recall'][i]:>10.3f}"
                     f"{m['f1'][i]:>10.3f}{m['support'][i]:>10d}")
    total = int(m["support"].sum())
    lines += ["",
              f"{'accuracy':<14}{'':>10}{'':>10}{m['accuracy']:>10.3f}{total:>10d}",
              f"{'macro avg':<14}{m['macro_precision']:>10.3f}{m['macro_recall']:>10.3f}"
              f"{m['macro_f1']:>10.3f}{total:>10d}"]
    return "\n".join(lines)
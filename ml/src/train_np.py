"""
Train the FelineHealth CNN written from scratch in NumPy (nn_layers.py).

Usage (from the FelineHealth-Ai project root, with venv active):
    python ml/src/prepare_data.py 64                      # once: makes ml/data/processed_64
    python ml/src/train_np_cnn.py --run_name np_v1

Outputs:
    ml/models/<run_name>.npz                 best weights (lowest validation loss)
    ml/results/<run_name>_history.csv        loss / accuracy per epoch
    ml/results/<run_name>_curves.png         training curves
    ml/results/<run_name>_val_report.txt     precision / recall / F1 per class
    ml/results/<run_name>_val_confusion.png  confusion matrix (validation set)
    ml/results/experiment_log_numpy.csv      one row per run
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
import nn_layers as nn          # noqa: E402
from np_metrics import classification_metrics, format_report  # noqa: E402

MODELS = Path("ml/models")
RESULTS = Path("ml/results")


# ----------------------------------------------------------------------------
# Data augmentation (NumPy only, training batches only)
# ----------------------------------------------------------------------------
def augment(xb, rng, shift_frac=0.1):
    """Random horizontal flip, random shift, brightness and contrast change.
    xb: float32 (N, H, W, C) in [0, 1]."""
    N, H, W, C = xb.shape
    out = xb.copy()
    flip = rng.random(N) < 0.5
    out[flip] = out[flip, :, ::-1, :]
    # random shift: reflect-pad then crop back to H x W
    p = int(round(H * shift_frac))
    if p > 0:
        padded = np.pad(out, ((0, 0), (p, p), (p, p), (0, 0)), mode="reflect")
        dy = rng.integers(0, 2 * p + 1, N)
        dx = rng.integers(0, 2 * p + 1, N)
        for i in range(N):
            out[i] = padded[i, dy[i]:dy[i] + H, dx[i]:dx[i] + W, :]
    # brightness (+/-0.1) and contrast (x0.9 - x1.1)
    bright = rng.uniform(-0.1, 0.1, (N, 1, 1, 1)).astype(np.float32)
    contrast = rng.uniform(0.9, 1.1, (N, 1, 1, 1)).astype(np.float32)
    mean = out.mean(axis=(1, 2, 3), keepdims=True)
    out = (out - mean) * contrast + mean + bright
    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def evaluate_loss_acc(model, X, y, batch_size=64):
    losses, correct = [], 0
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].astype(np.float32) / 255.0
        logits = model.forward(xb, training=False)
        loss, _, probs = nn.softmax_cross_entropy(logits, y[i:i + batch_size])
        losses.append(loss * len(xb))
        correct += int((probs.argmax(1) == y[i:i + batch_size]).sum())
    return sum(losses) / len(X), correct / len(X)


def save_plots(hist, metrics, class_names, run):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping plots")
        return
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_acc"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_acc"] for h in hist], label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[1].plot(ep, [h["val_loss"] for h in hist], label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(RESULTS / f"{run}_curves.png", dpi=110); plt.close()

    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix (acc {metrics['accuracy']:.1%})")
    plt.tight_layout(); plt.savefig(RESULTS / f"{run}_val_confusion.png", dpi=110); plt.close()


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="np_v1")
    p.add_argument("--data", default="ml/data/processed_64")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--filters", default="32,64,128")
    p.add_argument("--dense", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--no_bn", action="store_true")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--class_weights", action="store_true")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--start_from_epoch", type=int, default=5)
    p.add_argument("--max_train", type=int, default=0, help="debug: use only N images")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    data = Path(args.data)
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    X_train, y_train = np.load(data / "X_train.npy"), np.load(data / "y_train.npy")
    X_val, y_val = np.load(data / "X_val.npy"), np.load(data / "y_val.npy")
    if args.max_train:
        X_train, y_train = X_train[:args.max_train], y_train[:args.max_train]
    class_names = json.loads((data / "class_names.json").read_text())
    K = len(class_names)
    cw = None
    if args.class_weights:
        w = json.loads((data / "class_weights.json").read_text())
        cw = np.array([w[str(i)] for i in range(K)], dtype=np.float32)
    print(f"Train {X_train.shape}  Val {X_val.shape}  classes {class_names}")

    filters = tuple(int(f) for f in args.filters.split(","))
    model = nn.build_cnn(num_classes=K, filters=filters, dense_units=args.dense,
                         dropout=args.dropout, use_bn=not args.no_bn, seed=args.seed)
    opt = nn.Adam(model, lr=args.lr, weight_decay=args.weight_decay)
    print(f"Model parameters: {model.count_params():,}")

    best_loss, best_state, best_epoch, wait = np.inf, None, 0, 0
    lr_wait, hist = 0, []
    run = args.run_name
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(X_train))
        tl, tc = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            xb = X_train[idx].astype(np.float32) / 255.0
            if not args.no_augment:
                xb = augment(xb, rng)
            yb = y_train[idx]
            logits = model.forward(xb, training=True)
            loss, dlogits, probs = nn.softmax_cross_entropy(logits, yb, cw)
            model.backward(dlogits)
            opt.step()
            tl += loss * len(idx)
            tc += int((probs.argmax(1) == yb).sum())
        train_loss, train_acc = tl / len(order), tc / len(order)
        val_loss, val_acc = evaluate_loss_acc(model, X_val, y_val)
        hist.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                         val_loss=val_loss, val_acc=val_acc, lr=opt.lr))
        msg = ""
        if epoch >= args.start_from_epoch:
            if val_loss < best_loss - 1e-4:
                best_loss, best_state, best_epoch, wait, lr_wait = \
                    val_loss, model.state(), epoch, 0, 0
                msg = "  * best, saved"
            else:
                wait += 1; lr_wait += 1
                if lr_wait >= 6 and opt.lr > 1e-5:          # reduce LR on plateau
                    opt.lr = max(opt.lr * 0.5, 1e-5); lr_wait = 0
                    msg = f"  lr -> {opt.lr:.2e}"
        print(f"Epoch {epoch:3d}  loss {train_loss:.4f}  acc {train_acc:.3f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f}  "
              f"({time.time() - t0:.1f}s){msg}", flush=True)
        if wait >= args.patience:
            print(f"Early stopping (no improvement for {args.patience} epochs)")
            break

    if best_state is not None:
        model.load_state(best_state)
    model.save(MODELS / f"{run}.npz", filters=list(filters), dense=args.dense,
               dropout=args.dropout, use_bn=not args.no_bn,
               img_size=X_train.shape[1], class_names=class_names)

    with open(RESULTS / f"{run}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader(); w.writerows(hist)

    probs = model.predict_proba(X_val.astype(np.float32) / 255.0)
    m = classification_metrics(y_val, probs.argmax(1), K)
    report = format_report(m, class_names)
    (RESULTS / f"{run}_val_report.txt").write_text(report)
    save_plots(hist, m, class_names, run)

    log = RESULTS / "experiment_log_numpy.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "run_name", "img", "filters", "dense", "dropout", "bn",
                        "lr", "weight_decay", "augment", "class_weights", "seed",
                        "epochs_run", "best_epoch", "params", "val_acc", "val_macro_f1"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), run, X_train.shape[1],
                    args.filters, args.dense, args.dropout, not args.no_bn, args.lr,
                    args.weight_decay, not args.no_augment, args.class_weights, args.seed,
                    len(hist), best_epoch, model.count_params(),
                    round(m["accuracy"], 4), round(m["macro_f1"], 4)])

    print(f"\n=== Validation results (best epoch {best_epoch}) ===")
    print(report)
    print(f"\nSaved model: {MODELS / (run + '.npz')}")
    print(f"Val accuracy {m['accuracy']:.1%} | Val macro-F1 {m['macro_f1']:.3f}")


if __name__ == "__main__":
    main()

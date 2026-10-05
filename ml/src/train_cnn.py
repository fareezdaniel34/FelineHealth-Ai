import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.append(str(Path(__file__).parent))
from cnn_model import build_cnn  # noqa: E402

DATA = Path("ml/data/processed")
MODELS = Path("ml/models")
RESULTS = Path("ml/results")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_curves(history, path):
    import matplotlib.pyplot as plt
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def evaluate(model, X, y, class_names, prefix):
    """Accuracy, per-class precision/recall/F1 and confusion matrix."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    y_pred = model.predict(X, verbose=0).argmax(axis=1)
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    report = classification_report(y, y_pred, target_names=class_names, digits=3,
                                   zero_division=0)
    cm = confusion_matrix(y, y_pred)

    Path(f"{prefix}_report.txt").write_text(report)
    print(report)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix (acc {acc:.1%})")
    plt.tight_layout(); plt.savefig(f"{prefix}_confusion.png", dpi=110); plt.close()
    return acc, macro_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="cnn_v1")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--filters", default="32,64,128")
    p.add_argument("--dense", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="L2-style weight decay via AdamW, e.g. 1e-4 (0 = plain Adam)")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--no_class_weights", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default="ml/data/processed",
                   help="folder made by prepare_data.py, e.g. ml/data/processed_160")
    args = p.parse_args()
    DATA = Path(args.data)

    set_seed(args.seed)
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    X_train, y_train = np.load(DATA / "X_train.npy"), np.load(DATA / "y_train.npy")
    X_val, y_val = np.load(DATA / "X_val.npy"), np.load(DATA / "y_val.npy")
    class_names = json.loads((DATA / "class_names.json").read_text())
    class_weights = {int(k): v for k, v in
                     json.loads((DATA / "class_weights.json").read_text()).items()}
    print(f"Train {X_train.shape}, Val {X_val.shape}, classes {class_names}")

    filters = tuple(int(f) for f in args.filters.split(","))
    model = build_cnn(img_size=X_train.shape[1], num_classes=len(class_names),
                      filters=filters, dense_units=args.dense,
                      dropout=args.dropout, augment=not args.no_augment)
    model.summary()
    if args.weight_decay > 0:
        optimizer = keras.optimizers.AdamW(learning_rate=args.lr,
                                           weight_decay=args.weight_decay)
    else:
        optimizer = keras.optimizers.Adam(args.lr)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model_path = MODELS / f"{args.run_name}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                      start_from_epoch=10,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=8, min_lr=1e-5, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=None if args.no_class_weights else class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    plot_curves(history, RESULTS / f"{args.run_name}_curves.png")
    print("\n=== Validation results (best epoch) ===")
    val_acc, val_f1 = evaluate(model, X_val, y_val, class_names,
                               prefix=str(RESULTS / f"{args.run_name}_val"))

    # Experiment log: one row per run, used for the Chapter 4 table
    log = RESULTS / "experiment_log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "run_name", "filters", "dense", "dropout", "lr",
                        "weight_decay", "batch", "augment", "class_weights",
                        "epochs_run", "params", "val_acc", "val_macro_f1"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), args.run_name,
                    args.filters, args.dense, args.dropout, args.lr,
                    args.weight_decay, args.batch_size,
                    not args.no_augment, not args.no_class_weights,
                    len(history.history["loss"]), model.count_params(),
                    round(val_acc, 4), round(val_f1, 4)])
    print(f"\nSaved model: {model_path}")
    print(f"Val accuracy {val_acc:.1%} | Val macro-F1 {val_f1:.3f}")


if __name__ == "__main__":
    main()
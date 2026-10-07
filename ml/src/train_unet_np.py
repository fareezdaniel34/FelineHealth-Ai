"""
Train the NumPy U-Net (unet_np.py) on your lesion masks.

Usage (from the FelineHealth-Ai project root, with venv active):
    python ml/src/prepare_unet_data.py ml/data/masks_export 96     # once
    python ml/src/train_unet_np.py --run_name unet_v1
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
import nn_layers as nn          # noqa: E402
import unet_np as un            # noqa: E402

MODELS = Path("ml/models")
RESULTS = Path("ml/results")


def augment_pair(xb, mb, rng, shift_frac=0.1):
    """Same random flip + shift for image AND mask; brightness/contrast on image only."""
    N, H, W, _ = xb.shape
    x, m = xb.copy(), mb.copy()
    flip = rng.random(N) < 0.5
    x[flip] = x[flip, :, ::-1]
    m[flip] = m[flip, :, ::-1]
    p = int(round(H * shift_frac))
    if p > 0:
        xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode="reflect")
        mp = np.pad(m, ((0, 0), (p, p), (p, p), (0, 0)), mode="reflect")
        dy, dx = rng.integers(0, 2 * p + 1, N), rng.integers(0, 2 * p + 1, N)
        for i in range(N):
            x[i] = xp[i, dy[i]:dy[i] + H, dx[i]:dx[i] + W]
            m[i] = mp[i, dy[i]:dy[i] + H, dx[i]:dx[i] + W]
    bright = rng.uniform(-0.1, 0.1, (N, 1, 1, 1)).astype(np.float32)
    contrast = rng.uniform(0.9, 1.1, (N, 1, 1, 1)).astype(np.float32)
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    x = np.clip((x - mean) * contrast + mean + bright, 0.0, 1.0)
    return x, m


def evaluate(model, X, M, batch_size=16, threshold=0.5):
    """Validation loss + mean IoU / Dice (all images, and lesion images only)."""
    losses, ious, dices = [], [], []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].astype(np.float32) / 255.0
        mb = M[i:i + batch_size].astype(np.float32)
        logits = model.forward(xb, training=False)
        loss, _ = un.bce_dice_loss(logits, mb)
        losses.append(loss * len(xb))
        pred = un.sigmoid(logits)[..., 0] > threshold
        iou, dice = un.iou_dice_scores(pred, mb[..., 0])
        ious.append(iou); dices.append(dice)
    iou, dice = np.concatenate(ious), np.concatenate(dices)
    has_lesion = M.reshape(len(M), -1).max(1) > 0
    return dict(loss=sum(losses) / len(X), iou=iou.mean(), dice=dice.mean(),
                iou_lesion=iou[has_lesion].mean(), dice_lesion=dice[has_lesion].mean(),
                healthy_clean=float((iou[~has_lesion] == 1).mean()) if (~has_lesion).any() else 1.0)


def save_plots(hist, model, X, M, run):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping plots")
        return
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="validation")
    ax[0].set_title("Loss (BCE + Dice)"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep, [h["val_iou_lesion"] for h in hist], label="val IoU (lesion images)")
    ax[1].plot(ep, [h["val_dice_lesion"] for h in hist], label="val Dice (lesion images)")
    ax[1].set_title("Segmentation quality"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(RESULTS / f"{run}_curves.png", dpi=110); plt.close()

    n = min(8, len(X))
    pick = np.linspace(0, len(X) - 1, n).astype(int)
    prob = model.predict_mask_proba(X[pick].astype(np.float32) / 255.0)
    fig, ax = plt.subplots(3, n, figsize=(2 * n, 6.3))
    for j, i in enumerate(pick):
        ax[0, j].imshow(X[i]); ax[1, j].imshow(M[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        ax[2, j].imshow(prob[j] > 0.5, cmap="gray", vmin=0, vmax=1)
        for r in range(3):
            ax[r, j].axis("off")
    ax[0, 0].set_title("image", loc="left"); ax[1, 0].set_title("true mask", loc="left")
    ax[2, 0].set_title("predicted", loc="left")
    plt.tight_layout(); plt.savefig(RESULTS / f"{run}_val_predictions.png", dpi=100); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="unet_v1")
    p.add_argument("--data", default="ml/data/unet_96")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--base", type=int, default=16, help="filters in the first level")
    p.add_argument("--dice_weight", type=float, default=1.0)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--start_from_epoch", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    data = Path(args.data)
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    X_train, M_train = np.load(data / "X_train.npy"), np.load(data / "M_train.npy")
    X_val, M_val = np.load(data / "X_val.npy"), np.load(data / "M_val.npy")
    print(f"Train {X_train.shape}  Val {X_val.shape}")

    model = un.UNet(base=args.base, seed=args.seed)
    opt = nn.Adam(model, lr=args.lr, weight_decay=args.weight_decay)
    print(f"U-Net parameters: {model.count_params():,}")

    best_loss, best_state, best_epoch, wait, lr_wait, hist = np.inf, None, 0, 0, 0, []
    run = args.run_name
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(X_train))
        tl = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            xb = X_train[idx].astype(np.float32) / 255.0
            mb = M_train[idx].astype(np.float32)
            if not args.no_augment:
                xb, mb = augment_pair(xb, mb, rng)
            loss, dlogits = un.bce_dice_loss(model.forward(xb, training=True), mb,
                                             dice_weight=args.dice_weight)
            model.backward(dlogits)
            opt.step()
            tl += loss * len(idx)
        v = evaluate(model, X_val, M_val)
        hist.append(dict(epoch=epoch, train_loss=tl / len(order), val_loss=v["loss"],
                         val_iou=v["iou"], val_dice=v["dice"], val_iou_lesion=v["iou_lesion"],
                         val_dice_lesion=v["dice_lesion"], lr=opt.lr))
        msg = ""
        if epoch >= args.start_from_epoch:
            if v["loss"] < best_loss - 1e-4:
                best_loss, best_state, best_epoch, wait, lr_wait = v["loss"], model.state(), epoch, 0, 0
                msg = "  * best, saved"
            else:
                wait += 1; lr_wait += 1
                if lr_wait >= 8 and opt.lr > 1e-5:
                    opt.lr = max(opt.lr * 0.5, 1e-5); lr_wait = 0
                    msg = f"  lr -> {opt.lr:.2e}"
        print(f"Epoch {epoch:3d}  loss {tl / len(order):.4f}  val_loss {v['loss']:.4f}  "
              f"val IoU {v['iou_lesion']:.3f}  Dice {v['dice_lesion']:.3f} (lesion imgs)  "
              f"({time.time() - t0:.1f}s){msg}", flush=True)
        if wait >= args.patience:
            print(f"Early stopping (no improvement for {args.patience} epochs)")
            break

    if best_state is not None:
        model.load_state(best_state)
    model.save(MODELS / f"{run}.npz", base=args.base, img_size=X_train.shape[1])
    with open(RESULTS / f"{run}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader(); w.writerows(hist)
    v = evaluate(model, X_val, M_val)
    save_plots(hist, model, X_val, M_val, run)

    log = RESULTS / "experiment_log_unet.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "run_name", "img", "base", "lr", "weight_decay", "dice_weight",
                        "augment", "seed", "epochs_run", "best_epoch", "params",
                        "val_iou_lesion", "val_dice_lesion", "val_iou_all", "val_dice_all",
                        "healthy_clean"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), run, X_train.shape[1], args.base,
                    args.lr, args.weight_decay, args.dice_weight, not args.no_augment, args.seed,
                    len(hist), best_epoch, model.count_params(),
                    round(v["iou_lesion"], 4), round(v["dice_lesion"], 4),
                    round(v["iou"], 4), round(v["dice"], 4), round(v["healthy_clean"], 4)])

    print(f"\n=== Validation results (best epoch {best_epoch}) ===")
    print(f"Lesion images : IoU {v['iou_lesion']:.3f} | Dice {v['dice_lesion']:.3f}")
    print(f"All images    : IoU {v['iou']:.3f} | Dice {v['dice']:.3f}")
    print(f"Healthy images with no false lesion: {v['healthy_clean']:.0%}")
    print(f"Saved model: {MODELS / (run + '.npz')}")


if __name__ == "__main__":
    main()
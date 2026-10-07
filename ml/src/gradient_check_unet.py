"""
Gradient check for the U-Net parts (unet_np.py): Upsample2, BCE+Dice loss,
and the full U-Net with skip connections.

Usage:  python ml/src/gradient_check_unet.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
import nn_layers as nn          # noqa: E402
import unet_np as un            # noqa: E402

rng = np.random.default_rng(0)


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def numerical_grad(f, x, h=1e-5, max_checks=30):
    idx = [tuple(rng.integers(0, s) for s in x.shape) for _ in range(max_checks)]
    num = []
    for i in idx:
        old = x[i]
        x[i] = old + h; fp = f()
        x[i] = old - h; fm = f()
        x[i] = old
        num.append((fp - fm) / (2 * h))
    return idx, np.array(num)


def report(name, worst):
    status = "OK " if worst < 1e-5 else "FAIL"
    print(f"[{status}] {name:<34} max relative error = {worst:.2e}")
    return worst < 1e-5


def check_upsample():
    up = un.Upsample2()
    x = rng.standard_normal((2, 3, 3, 2))
    R = rng.standard_normal((2, 6, 6, 2))
    f = lambda: float((up.forward(x, True) * R).sum())
    f(); dx = up.backward(R)
    idx, num = numerical_grad(f, x)
    return report("Upsample2 (nearest x2)", rel_error(np.array([dx[i] for i in idx]), num))


def check_loss():
    logits = rng.standard_normal((3, 5, 5, 1))
    mask = (rng.random((3, 5, 5, 1)) < 0.3).astype(np.float64)
    mask[2] = 0                                        # one "healthy" (empty) mask
    f = lambda: un.bce_dice_loss(logits, mask)[0]
    _, d = un.bce_dice_loss(logits, mask)
    idx, num = numerical_grad(f, logits)
    return report("BCE + Dice loss", rel_error(np.array([d[i] for i in idx]), num))


def check_full_unet():
    model = un.UNet(base=2, seed=3)
    for l in model.layers:
        for k in l.params:
            l.params[k] = l.params[k].astype(np.float64)
        if isinstance(l, nn.BatchNorm):
            l.running_mean = l.running_mean.astype(np.float64)
            l.running_var = l.running_var.astype(np.float64)
    X = rng.standard_normal((3, 16, 16, 3))
    M = (rng.random((3, 16, 16, 1)) < 0.3).astype(np.float64)

    def f():
        return un.bce_dice_loss(model.forward(X, training=True), M)[0]

    _, d = un.bce_dice_loss(model.forward(X, training=True), M)
    dX = model.backward(d)
    worst = 0.0
    grads = {(i, k): l.grads[k].copy() for i, l in enumerate(model.layers) for k in l.params}
    for (i, k), g in grads.items():
        idx, num = numerical_grad(f, model.layers[i].params[k], max_checks=6)
        worst = max(worst, rel_error(np.array([g[j] for j in idx]), num))
    idx, num = numerical_grad(f, X, max_checks=20)
    worst = max(worst, rel_error(np.array([dX[j] for j in idx]), num))
    return report("FULL U-Net (skips) + loss", worst)


if __name__ == "__main__":
    print("U-Net gradient check (analytic backprop vs numerical gradient)\n")
    results = [check_upsample(), check_loss(), check_full_unet()]
    print("\nALL PASSED" if all(results) else "\nSOME CHECKS FAILED")
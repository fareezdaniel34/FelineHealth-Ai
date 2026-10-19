"""
Gradient check: proves that every backward() in nn_layers.py is correct.

Compares the analytic gradient (our backward() code) with the numerical
gradient (f(x+h) - f(x-h)) / 2h. Relative error < 1e-5 means the maths is right.

Usage:  python ml/src/gradient_check.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
import nn_layers as nn  # noqa: E402

rng = np.random.default_rng(0)


def to64(layer):
    for k in layer.params:
        layer.params[k] = layer.params[k].astype(np.float64)
    if isinstance(layer, nn.BatchNorm):
        layer.running_mean = layer.running_mean.astype(np.float64)
        layer.running_var = layer.running_var.astype(np.float64)
    return layer


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def numerical_grad(f, x, h=1e-5, max_checks=40):
    """Numerical gradient at up to max_checks random positions of x."""
    idx = [tuple(rng.integers(0, s) for s in x.shape) for _ in range(max_checks)]
    num = []
    for i in idx:
        old = x[i]
        x[i] = old + h; fp = f()
        x[i] = old - h; fm = f()
        x[i] = old
        num.append((fp - fm) / (2 * h))
    return idx, np.array(num)


def check_layer(name, layer, x_shape, training=True, seed_dropout=False):
    to64(layer)
    x = rng.standard_normal(x_shape)
    out = layer.forward(x, training)
    R = rng.standard_normal(out.shape)          # random "upstream gradient"

    def f():
        if seed_dropout:
            layer.rng = np.random.default_rng(123)
        return float((layer.forward(x, training) * R).sum())

    f()
    dx = layer.backward(R)
    worst = 0.0
    idx, num = numerical_grad(f, x)
    worst = max(worst, rel_error(np.array([dx[i] for i in idx]), num))
    for pname, p in layer.params.items():
        f(); layer.backward(R)
        analytic = layer.grads[pname].copy()
        idx, num = numerical_grad(f, p)
        worst = max(worst, rel_error(np.array([analytic[i] for i in idx]), num))
    status = "OK " if worst < 1e-5 else "FAIL"
    print(f"[{status}] {name:<28} max relative error = {worst:.2e}")
    return worst < 1e-5


def check_full_model():
    model = nn.build_cnn(num_classes=4, filters=(4, 6), dense_units=5,
                         dropout=0.0, use_bn=True, seed=1)
    for l in model.layers:
        to64(l)
    X = rng.standard_normal((3, 8, 8, 3))
    y = np.array([0, 2, 3])
    cw = np.array([1.0, 1.3, 0.8, 1.1])

    def f():
        return nn.softmax_cross_entropy(model.forward(X, training=True), y, cw)[0]

    loss, dlogits, _ = nn.softmax_cross_entropy(model.forward(X, training=True), y, cw)
    model.backward(dlogits)
    worst = 0.0
    for layer in model.layers:
        for pname, p in layer.params.items():
            analytic = layer.grads[pname].copy()
            idx, num = numerical_grad(f, p, max_checks=15)
            worst = max(worst, rel_error(np.array([analytic[i] for i in idx]), num))
    status = "OK " if worst < 1e-5 else "FAIL"
    print(f"[{status}] {'FULL CNN + softmax loss':<28} max relative error = {worst:.2e}")
    return worst < 1e-5


if __name__ == "__main__":
    print("Gradient check (analytic backprop vs numerical gradient)\n")
    results = [
        check_layer("Conv2D 3x3 (with bias)", nn.Conv2D(3, 4, rng=rng), (2, 6, 6, 3)),
        check_layer("Conv2D 3x3 (no bias)", nn.Conv2D(2, 3, use_bias=False, rng=rng), (2, 5, 5, 2)),
        check_layer("BatchNorm (training)", nn.BatchNorm(3), (4, 3, 3, 3)),
        check_layer("ReLU", nn.ReLU(), (2, 4, 4, 3)),
        check_layer("MaxPool 2x2", nn.MaxPool2(), (2, 4, 4, 3)),
        check_layer("GlobalAvgPool", nn.GlobalAvgPool(), (2, 4, 4, 3)),
        check_layer("Dense", nn.Dense(5, 4, rng=rng), (3, 5)),
        check_layer("Dropout (fixed mask)", nn.Dropout(0.5), (3, 6), seed_dropout=True),
        check_full_model(),
    ]
    print("\nALL PASSED" if all(results) else "\nSOME CHECKS FAILED")
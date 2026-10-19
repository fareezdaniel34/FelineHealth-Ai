"""
FelineHealth-AI: neural network layers written from scratch with NumPy.

Only NumPy is used, for basic array arithmetic. Every forward pass, backward
pass (backpropagation) and optimiser update is implemented here by hand.

Data layout: images are (N, H, W, C)  = (batch, height, width, channels).
"""
import numpy as np

DTYPE = np.float32


# ----------------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------------
class Layer:
    """Every layer has forward(), backward() and (optionally) parameters."""

    def __init__(self):
        self.params = {}   # name -> array (learned values)
        self.grads = {}    # name -> array (gradient of loss w.r.t. param)

    def forward(self, x, training):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Convolution (3x3, stride 1, 'same' padding) using im2col
# ----------------------------------------------------------------------------
class Conv2D(Layer):
    """
    out[n,i,j,f] = sum_{a,b,c} x_pad[n,i+a,j+b,c] * W[f,c,a,b] + bias[f]

    im2col: every k x k patch of the input becomes one row of a matrix, so the
    whole convolution becomes ONE matrix multiplication (fast in NumPy).
    """

    def __init__(self, in_ch, out_ch, k=3, use_bias=True, rng=None):
        super().__init__()
        rng = rng or np.random.default_rng()
        self.k, self.pad = k, k // 2
        fan_in = in_ch * k * k
        # He initialisation (good for ReLU networks)
        self.params["W"] = (rng.standard_normal((out_ch, in_ch, k, k))
                            * np.sqrt(2.0 / fan_in)).astype(DTYPE)
        self.use_bias = use_bias
        if use_bias:
            self.params["b"] = np.zeros(out_ch, dtype=DTYPE)

    def forward(self, x, training):
        N, H, W, C = x.shape
        k, p = self.k, self.pad
        xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
        # windows: (N, H, W, C, k, k) -> rows of length C*k*k
        win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(1, 2))
        cols = win.reshape(N * H * W, C * k * k)
        Wm = self.params["W"].reshape(self.params["W"].shape[0], -1)   # (F, C*k*k)
        out = cols @ Wm.T
        if self.use_bias:
            out += self.params["b"]
        self.cache = (x.shape, cols)
        return out.reshape(N, H, W, -1)

    def backward(self, dout):
        (N, H, W, C), cols = self.cache
        k, p = self.k, self.pad
        F = dout.shape[-1]
        d = dout.reshape(-1, F)                                         # (N*H*W, F)
        Wm = self.params["W"].reshape(F, -1)
        # dW = d^T @ cols, computed in row chunks (same result, much faster on
        # some NumPy builds because the chunks fit in the CPU cache)
        dW = np.zeros((F, cols.shape[1]), dtype=cols.dtype)
        for s in range(0, d.shape[0], 2048):
            dW += d[s:s + 2048].T @ cols[s:s + 2048]
        self.grads["W"] = dW.reshape(self.params["W"].shape)
        if self.use_bias:
            self.grads["b"] = d.sum(axis=0)
        dcols = (d @ Wm).reshape(N, H, W, C, k, k)
        # col2im: add each patch gradient back to where it came from
        dxp = np.zeros((N, H + 2 * p, W + 2 * p, C), dtype=dout.dtype)
        for a in range(k):
            for b in range(k):
                dxp[:, a:a + H, b:b + W, :] += dcols[..., a, b]
        return dxp[:, p:p + H, p:p + W, :]


# ----------------------------------------------------------------------------
# Batch normalisation (per channel)
# ----------------------------------------------------------------------------
class BatchNorm(Layer):
    def __init__(self, ch, momentum=0.9, eps=1e-5):
        super().__init__()
        self.params["gamma"] = np.ones(ch, dtype=DTYPE)
        self.params["beta"] = np.zeros(ch, dtype=DTYPE)
        self.running_mean = np.zeros(ch, dtype=DTYPE)
        self.running_var = np.ones(ch, dtype=DTYPE)
        self.momentum, self.eps = momentum, eps

    def forward(self, x, training):
        axes = tuple(range(x.ndim - 1))           # all except channel
        if training:
            mu = x.mean(axis=axes)
            var = x.var(axis=axes)
            m = self.momentum
            self.running_mean = m * self.running_mean + (1 - m) * mu
            self.running_var = m * self.running_var + (1 - m) * var
        else:
            mu, var = self.running_mean, self.running_var
        inv_std = 1.0 / np.sqrt(var + self.eps)
        xhat = (x - mu) * inv_std
        self.cache = (xhat, inv_std, axes)
        return self.params["gamma"] * xhat + self.params["beta"]

    def backward(self, dout):
        xhat, inv_std, axes = self.cache
        # float (not numpy int) so float32 stays float32 (int64 would upcast to float64)
        M = float(np.prod([dout.shape[a] for a in axes]))
        self.grads["gamma"] = (dout * xhat).sum(axis=axes)
        self.grads["beta"] = dout.sum(axis=axes)
        dxhat = dout * self.params["gamma"]
        # standard batch-norm backward formula
        dx = (inv_std / M) * (M * dxhat
                              - dxhat.sum(axis=axes)
                              - xhat * (dxhat * xhat).sum(axis=axes))
        return dx.astype(dout.dtype, copy=False)


# ----------------------------------------------------------------------------
# Simple layers
# ----------------------------------------------------------------------------
class ReLU(Layer):
    def forward(self, x, training):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2(Layer):
    """2x2 max pooling, stride 2 (H and W must be even)."""

    def forward(self, x, training):
        N, H, W, C = x.shape
        r = x.reshape(N, H // 2, 2, W // 2, 2, C)
        out = r.max(axis=(2, 4))
        self.mask = (r == out[:, :, None, :, None, :])
        self.shape = x.shape
        return out

    def backward(self, dout):
        d = self.mask * dout[:, :, None, :, None, :]
        return d.reshape(self.shape)


class GlobalAvgPool(Layer):
    """(N, H, W, C) -> (N, C): average of each feature map."""

    def forward(self, x, training):
        self.shape = x.shape
        return x.mean(axis=(1, 2))

    def backward(self, dout):
        N, H, W, C = self.shape
        return np.broadcast_to(dout[:, None, None, :] / (H * W), self.shape).copy()


class Dense(Layer):
    def __init__(self, in_dim, out_dim, rng=None, relu_init=True):
        super().__init__()
        rng = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / in_dim) if relu_init else np.sqrt(1.0 / in_dim)
        self.params["W"] = (rng.standard_normal((in_dim, out_dim)) * scale).astype(DTYPE)
        self.params["b"] = np.zeros(out_dim, dtype=DTYPE)

    def forward(self, x, training):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"] = self.x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T


class Dropout(Layer):
    """Inverted dropout: active only during training."""

    def __init__(self, rate, rng=None):
        super().__init__()
        self.rate = rate
        self.rng = rng or np.random.default_rng()

    def forward(self, x, training):
        if not training or self.rate == 0:
            self.mask = None
            return x
        keep = 1.0 - self.rate
        self.mask = (self.rng.random(x.shape) < keep).astype(x.dtype) / keep
        return x * self.mask

    def backward(self, dout):
        return dout if self.mask is None else dout * self.mask


# ----------------------------------------------------------------------------
# Loss: softmax + cross-entropy (combined, numerically stable)
# ----------------------------------------------------------------------------
def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, y, class_weights=None):
    """Returns (mean loss, gradient w.r.t. logits, probabilities)."""
    N = logits.shape[0]
    probs = softmax(logits)
    w = np.ones(N, dtype=logits.dtype) if class_weights is None else class_weights[y]
    loss = -(w * np.log(probs[np.arange(N), y] + 1e-12)).sum() / N
    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1
    dlogits *= w[:, None] / N
    return loss, dlogits, probs


# ----------------------------------------------------------------------------
# Model container
# ----------------------------------------------------------------------------
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def predict_proba(self, X, batch_size=64):
        out = []
        for i in range(0, len(X), batch_size):
            out.append(softmax(self.forward(X[i:i + batch_size], training=False)))
        return np.concatenate(out)

    def count_params(self):
        return int(sum(p.size for l in self.layers for p in l.params.values()))

    # --- saving / loading (weights + batch-norm running statistics) ---
    def state(self):
        s = {}
        for i, l in enumerate(self.layers):
            for k, v in l.params.items():
                s[f"{i}_{k}"] = v.copy()
            if isinstance(l, BatchNorm):
                s[f"{i}_running_mean"] = l.running_mean.copy()
                s[f"{i}_running_var"] = l.running_var.copy()
        return s

    def load_state(self, s):
        for i, l in enumerate(self.layers):
            for k in l.params:
                l.params[k] = s[f"{i}_{k}"].astype(DTYPE)
            if isinstance(l, BatchNorm):
                l.running_mean = s[f"{i}_running_mean"].astype(DTYPE)
                l.running_var = s[f"{i}_running_var"].astype(DTYPE)

    def save(self, path, **extra):
        np.savez(path, **self.state(), **{f"meta_{k}": np.array(v) for k, v in extra.items()})

    def load(self, path):
        d = np.load(path)
        self.load_state({k: d[k] for k in d.files if not k.startswith("meta_")})


# ----------------------------------------------------------------------------
# Optimiser: Adam / AdamW (decoupled weight decay)
# ----------------------------------------------------------------------------
class Adam:
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        self.model, self.lr = model, lr
        self.b1, self.b2, self.eps, self.wd = beta1, beta2, eps, weight_decay
        self.t = 0
        self.m, self.v = {}, {}

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.model.layers):
            for name, p in layer.params.items():
                g = layer.grads[name]
                key = (i, name)
                if key not in self.m:
                    self.m[key] = np.zeros_like(p)
                    self.v[key] = np.zeros_like(p)
                self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
                self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * g * g
                m_hat = self.m[key] / (1 - self.b1 ** self.t)
                v_hat = self.v[key] / (1 - self.b2 ** self.t)
                update = m_hat / (np.sqrt(v_hat) + self.eps)
                # weight decay only on weight matrices, not on biases / BN
                if self.wd > 0 and name == "W":
                    update = update + self.wd * p
                p -= (self.lr * update).astype(p.dtype)


# ----------------------------------------------------------------------------
# The FelineHealth CNN (same design as the earlier Keras reference model)
# ----------------------------------------------------------------------------
def build_cnn(num_classes=4, filters=(32, 64, 128), dense_units=128,
              dropout=0.5, use_bn=True, seed=42):
    rng = np.random.default_rng(seed)
    layers, in_ch = [], 3
    for f in filters:
        layers.append(Conv2D(in_ch, f, k=3, use_bias=not use_bn, rng=rng))
        if use_bn:
            layers.append(BatchNorm(f))
        layers += [ReLU(), MaxPool2()]
        in_ch = f
    layers += [GlobalAvgPool(),
               Dense(in_ch, dense_units, rng=rng), ReLU(),
               Dropout(dropout, rng=rng),
               Dense(dense_units, num_classes, rng=rng, relu_init=False)]
    return Sequential(layers)
"""
FelineHealth-AI: U-Net for lesion segmentation, written from scratch with NumPy.

Re-uses Conv2D, BatchNorm, ReLU, MaxPool2 and Adam from nn_layers.py and adds:
  - Upsample2        (nearest-neighbour x2, for the decoder)
  - skip connections (concatenate encoder + decoder feature maps)
  - bce_dice_loss    (binary cross-entropy + Dice loss on a sigmoid output)
  - IoU / Dice metrics
"""
import numpy as np

from nn_layers import BatchNorm, Conv2D, MaxPool2, ReLU, Sequential


class Upsample2:
    """Nearest-neighbour upsampling x2: each pixel becomes a 2x2 block."""

    def __init__(self):
        self.params, self.grads = {}, {}

    def forward(self, x, training):
        return x.repeat(2, axis=1).repeat(2, axis=2)

    def backward(self, dout):
        N, H, W, C = dout.shape
        # each input pixel was copied to 4 output pixels -> sum their gradients
        return dout.reshape(N, H // 2, 2, W // 2, 2, C).sum(axis=(2, 4))


def conv_bn_relu(in_ch, out_ch, rng):
    return [Conv2D(in_ch, out_ch, 3, use_bias=False, rng=rng), BatchNorm(out_ch), ReLU()]


class UNet(Sequential):
    """
    Encoder (contracting path)            Decoder (expanding path)
    enc1: 2x conv  (H)      ---skip1--->  dec1: up + concat + 2x conv -> 1x1 conv -> mask
    enc2: 2x conv  (H/2)    ---skip2--->  dec2
    enc3: 2x conv  (H/4)    ---skip3--->  dec3
    bottleneck: 2x conv (H/8)
    Input H and W must be divisible by 8.
    """

    def __init__(self, base=16, seed=42):
        rng = np.random.default_rng(seed)
        b = base
        self.enc = [conv_bn_relu(3, b, rng) + conv_bn_relu(b, b, rng),
                    conv_bn_relu(b, 2 * b, rng) + conv_bn_relu(2 * b, 2 * b, rng),
                    conv_bn_relu(2 * b, 4 * b, rng) + conv_bn_relu(4 * b, 4 * b, rng)]
        self.pools = [MaxPool2() for _ in range(3)]
        self.bottleneck = conv_bn_relu(4 * b, 8 * b, rng) + conv_bn_relu(8 * b, 8 * b, rng)
        self.ups = [Upsample2() for _ in range(3)]
        # decoder input channels = upsampled channels + skip channels
        self.dec = [conv_bn_relu(8 * b + 4 * b, 4 * b, rng) + conv_bn_relu(4 * b, 4 * b, rng),
                    conv_bn_relu(4 * b + 2 * b, 2 * b, rng) + conv_bn_relu(2 * b, 2 * b, rng),
                    conv_bn_relu(2 * b + b, b, rng) + conv_bn_relu(b, b, rng)]
        self.head = Conv2D(b, 1, k=1, use_bias=True, rng=rng)
        self.head.params["W"] *= 0.5 ** 0.5    # ~Xavier scale for the sigmoid output
        # flat list of every layer (used by Adam and save/load in Sequential)
        self.layers = ([l for blk in self.enc for l in blk] + self.pools + self.bottleneck
                       + self.ups + [l for blk in self.dec for l in blk] + [self.head])

    @staticmethod
    def _run(block, x, training):
        for l in block:
            x = l.forward(x, training)
        return x

    @staticmethod
    def _back(block, d):
        for l in reversed(block):
            d = l.backward(d)
        return d

    def forward(self, x, training=False):
        skips = []
        for blk, pool in zip(self.enc, self.pools):
            x = self._run(blk, x, training)
            skips.append(x)
            x = pool.forward(x, training)
        x = self._run(self.bottleneck, x, training)
        self.split_ch = []
        for blk, up, skip in zip(self.dec, self.ups, reversed(skips)):
            x = up.forward(x, training)
            self.split_ch.append(x.shape[-1])
            x = np.concatenate([x, skip], axis=-1)          # skip connection
            x = self._run(blk, x, training)
        return self.head.forward(x, training)                 # logits (N, H, W, 1)

    def backward(self, dout):
        d = self.head.backward(dout)
        dskips = []
        for blk, up, c in zip(reversed(self.dec), reversed(self.ups), reversed(self.split_ch)):
            d = self._back(blk, d)
            d_up, d_skip = d[..., :c], d[..., c:]              # undo the concatenation
            dskips.append(d_skip)
            d = up.backward(d_up)
        d = self._back(self.bottleneck, d)
        # dskips was filled shallow -> deep; the encoder is walked deep -> shallow
        for blk, pool, ds in zip(reversed(self.enc), reversed(self.pools), reversed(dskips)):
            d = pool.backward(d) + ds                          # gradient from both paths
            d = self._back(blk, d)
        return d

    def predict_mask_proba(self, X, batch_size=16):
        out = []
        for i in range(0, len(X), batch_size):
            out.append(sigmoid(self.forward(X[i:i + batch_size], training=False)))
        return np.concatenate(out)[..., 0]


# ----------------------------------------------------------------------------
# Loss and metrics
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def bce_dice_loss(logits, mask, dice_weight=1.0, smooth=1.0):
    """
    logits, mask: (N, H, W, 1). mask is 0/1.
    BCE  = mean[ -t*log(p) - (1-t)*log(1-p) ]                  (over all pixels)
    Dice = 1 - (2*sum(p*t) + s) / (sum(p) + sum(t) + s)         (over the whole batch)
    Batch-level Dice: healthy images (empty masks) only add false-positive
    pixels to sum(p) instead of getting an unfair per-image Dice penalty.
    Returns (loss, dloss/dlogits).
    """
    t = mask.astype(logits.dtype)
    p = sigmoid(logits)
    n_pix = p.size
    # BCE written in a numerically stable form using the logits
    bce = (np.maximum(logits, 0) - logits * t + np.log1p(np.exp(-np.abs(logits)))).mean()
    d_bce = (p - t) / n_pix

    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    dice = (2 * inter + smooth) / (denom + smooth)
    dice_loss = 1 - dice
    # d dice / d p, then chain rule through the sigmoid: dp/dz = p(1-p)
    d_dice_dp = (2 * t * (denom + smooth) - (2 * inter + smooth)) / (denom + smooth) ** 2
    d_dice = -d_dice_dp * p * (1 - p)

    loss = bce + dice_weight * dice_loss
    return float(loss), (d_bce + dice_weight * d_dice).astype(logits.dtype)


def iou_dice_scores(pred_mask, true_mask):
    """Per-image IoU and Dice for binary masks (N, H, W).
    If both masks are empty (healthy image, nothing predicted) the score is 1."""
    p = pred_mask.astype(bool)
    t = true_mask.astype(bool)
    axes = tuple(range(1, p.ndim))
    inter = (p & t).sum(axis=axes)
    union = (p | t).sum(axis=axes)
    psum, tsum = p.sum(axis=axes), t.sum(axis=axes)
    iou = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    dice = np.where(psum + tsum == 0, 1.0, 2 * inter / np.maximum(psum + tsum, 1))
    return iou, dice
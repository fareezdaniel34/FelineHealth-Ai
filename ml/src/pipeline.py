"""
FelineHealth-AI full pipeline (NumPy models only):

    photo -> U-Net (letterboxed) -> lesion mask -> crop around lesion -> CNN -> disease

Used by evaluate_pipeline.py and later by the Flask backend.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent))
import nn_layers as nn                                   # noqa: E402
import unet_np as un                                     # noqa: E402
from tune_unet_postprocess import postprocess            # noqa: E402


# ----------------------------------------------------------------------------
# Image helpers (must match the training preprocessing)
# ----------------------------------------------------------------------------
def centre_crop_resize(img, size):
    """CNN preprocessing (same as prepare_data.py)."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.BILINEAR)


def letterbox(img, size):
    """U-Net preprocessing (same as prepare_unet_data.py). Returns image + geometry."""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), 0)
    off = ((s - w) // 2, (s - h) // 2)
    canvas.paste(img, off)
    return canvas.resize((size, size), Image.BILINEAR), (s, off)


def mask_to_original(mask_small, geom, orig_size):
    """Map a U-Net mask (size x size) back onto the original photo."""
    s, (ox, oy) = geom
    w, h = orig_size
    big = Image.fromarray(mask_small.astype(np.uint8) * 255).resize((s, s), Image.NEAREST)
    return np.asarray(big)[oy:oy + h, ox:ox + w] > 0


def lesion_box(mask, margin=0.25, min_frac=0.15):
    """Square box around all lesion pixels, enlarged by `margin` on each side and at
    least `min_frac` of the photo's short side (so tiny lesions keep some context)."""
    ys, xs = np.where(mask)
    H, W = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0)
    side = int(max(side * (1 + 2 * margin), min_frac * min(H, W)))
    side = min(side, H, W)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    top = int(np.clip(cy - side // 2, 0, H - side))
    left = int(np.clip(cx - side // 2, 0, W - side))
    return left, top, left + side, top + side


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def load_cnn(path):
    d = np.load(path)
    filters = tuple(int(f) for f in d["meta_filters"])
    model = nn.build_cnn(num_classes=len(d["meta_class_names"]), filters=filters,
                         dense_units=int(d["meta_dense"]), dropout=float(d["meta_dropout"]),
                         use_bn=bool(d["meta_use_bn"]))
    model.load(path)
    return model, int(d["meta_img_size"]), [str(c) for c in d["meta_class_names"]]


def load_unet(path, postprocess_json):
    d = np.load(path)
    model = un.UNet(base=int(d["meta_base"]))
    model.load(path)
    pp = json.load(open(postprocess_json))
    return model, int(d["meta_img_size"]), pp


class FelineHealthPipeline:
    def __init__(self, cnn_path, unet_path, postprocess_json, crop_cnn_path=None):
        self.cnn, self.cnn_size, self.class_names = load_cnn(cnn_path)
        self.unet, self.unet_size, self.pp = load_unet(unet_path, postprocess_json)
        # optional second CNN trained on lesion crops (make_crop_dataset.py)
        if crop_cnn_path:
            self.crop_cnn, self.crop_size, _ = load_cnn(crop_cnn_path)
        else:
            self.crop_cnn, self.crop_size = self.cnn, self.cnn_size

    def cnn_probs(self, img, crop_model=False):
        model, size = (self.crop_cnn, self.crop_size) if crop_model else (self.cnn, self.cnn_size)
        x = np.asarray(centre_crop_resize(img, size), dtype=np.float32)[None] / 255.0
        return model.predict_proba(x)[0]

    def segment(self, img):
        small, geom = letterbox(img, self.unet_size)
        x = np.asarray(small, dtype=np.float32)[None] / 255.0
        prob = self.unet.predict_mask_proba(x)[0]
        mask_small = postprocess(prob, self.pp["threshold"], self.pp["min_blob_px"])
        return mask_to_original(mask_small, geom, img.size)

    def predict(self, img):
        """img: PIL image. Returns probabilities for 3 strategies + mask + box."""
        img = img.convert("RGB")
        p_full = self.cnn_probs(img)
        mask = self.segment(img)
        if mask.any():
            box = lesion_box(mask)
            p_crop = self.cnn_probs(img.crop(box), crop_model=True)
        else:                                   # no lesion found -> use the full photo
            box, p_crop = None, self.cnn_probs(img, crop_model=True)
        p_comb = (p_full + p_crop) / 2
        return {"full": p_full, "crop": p_crop, "combined": p_comb,
                "mask": mask, "box": box, "lesion_found": bool(mask.any()),
                "lesion_area_pct": float(mask.mean() * 100)}
"""
FelineHealth-AI inference service: turns an uploaded photo into the JSON the
React app shows. Uses ONLY our own NumPy models (pipeline.py). No Flask code
here, so it can be tested on its own.
"""
import base64
import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent            # FelineHealth-Ai/
sys.path.append(str(ROOT / "ml" / "src"))
from pipeline import FelineHealthPipeline  # noqa: E402

MODELS = ROOT / "ml" / "models"
CONFIG = {
    "cnn": MODELS / "np_v2_96.npz",          # CNN on the full photo
    "crop_cnn": MODELS / "np_crop_96.npz",   # CNN on the U-Net lesion crop
    "unet": MODELS / "unet_v1.npz",
    "postprocess": MODELS / "unet_postprocess.json",
    "min_confidence": 0.45,                  # below this -> "uncertain" (calibrated on val)
    # disease predicted but U-Net found no lesion -> "uncertain"
    # (gate_rule_analysis.py on val, pre-specified rule MAX_LOST=2 -> gate "always":
    #  removes 3 wrong disease results, loses 2 correct ones, health false alarms 6 -> 4)
    "empty_mask_gate": True,
    "max_side": 1024,                        # downscale huge photos (speed)
}

DISCLAIMER = ("FelineHealth-AI is a preliminary screening tool, not a diagnosis. "
              "Please consult a veterinarian to confirm any skin condition.")

DISEASE_INFO = {
    "Ringworm": {
        "name": "Ringworm (Dermatophytosis)",
        "description": "A fungal infection of the skin and hair. It often appears as round "
                       "patches of hair loss with scaly or crusty skin, commonly on the head, "
                       "ears and legs.",
        "contagious": "Yes. It can spread to other pets and to humans.",
        "advice": "Visit a vet for confirmation (e.g. fungal culture). Wash hands after "
                  "handling the cat and clean bedding and grooming tools.",
    },
    "Flea Allergy": {
        "name": "Flea Allergy Dermatitis",
        "description": "An allergic reaction to flea saliva. Signs include intense "
                       "itching, small scabs or red bumps, and hair loss, often near the "
                       "lower back, tail base and neck.",
        "contagious": "The allergy itself is not contagious, but fleas spread easily "
                      "between pets.",
        "advice": "Ask a vet about safe flea treatment for all pets in the home and treat "
                  "the environment (bedding, carpets).",
    },
    "Scabies": {
        "name": "Feline Scabies (Notoedric Mange)",
        "description": "A skin disease caused by mites. It causes severe itching and "
                       "thick crusts, typically starting on the ears and face.",
        "contagious": "Yes. Highly contagious between cats; it can cause temporary itching "
                      "in humans.",
        "advice": "See a vet promptly for anti-parasite treatment and keep the cat "
                  "separated from other cats until treated.",
    },
    "Health": {
        "name": "No skin disease detected",
        "description": "The model did not find signs of ringworm, flea allergy or scabies "
                       "in this photo.",
        "contagious": "-",
        "advice": "If you still notice itching, hair loss or redness, take a clear, close-up "
                  "photo of the area or consult a vet.",
    },
}


def _jpeg_data_url(img, quality=85):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def make_overlay(img, mask, box):
    """Photo with the lesion area tinted red and the crop box outlined."""
    base = img.convert("RGBA")
    red = Image.new("RGBA", base.size, (255, 40, 40, 0))
    alpha = Image.fromarray((mask * 110).astype(np.uint8))       # semi-transparent
    red.putalpha(alpha)
    out = Image.alpha_composite(base, red)
    if box is not None:
        ImageDraw.Draw(out).rectangle(box, outline=(255, 215, 0, 255),
                                      width=max(2, min(img.size) // 150))
    return out.convert("RGB")


class FelineHealthService:
    def __init__(self, config=None):
        self.cfg = dict(CONFIG, **(config or {}))
        missing = [str(self.cfg[k]) for k in ("cnn", "crop_cnn", "unet", "postprocess")
                   if not Path(self.cfg[k]).exists()]
        if missing:
            raise FileNotFoundError(f"Model files not found: {missing}")
        self.pipe = FelineHealthPipeline(str(self.cfg["cnn"]), str(self.cfg["unet"]),
                                         str(self.cfg["postprocess"]), str(self.cfg["crop_cnn"]))
        self.class_names = self.pipe.class_names
        # our NumPy layers store intermediate values (caches) while running,
        # so only one prediction may run at a time
        self.lock = threading.Lock()

    def analyse(self, image_bytes):
        t0 = time.time()
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")       # fix phone rotation
        img.thumbnail((self.cfg["max_side"], self.cfg["max_side"]))
        with self.lock:
            out = self.pipe.predict(img)

        probs = out["combined"]                                 # strategy C
        order = np.argsort(probs)[::-1]
        k, k2 = int(order[0]), int(order[1])
        label, conf = self.class_names[k], float(probs[k])
        second = self.class_names[k2]
        reason = None                     # why a result is "uncertain"

        if conf < self.cfg["min_confidence"]:
            result_type, reason = "uncertain", "low_confidence"
            message = (f"The result is unclear. Most likely: {label} ({conf:.0%}) or "
                       f"{second} ({float(probs[k2]):.0%}). Try a clear, well-lit, close-up "
                       f"photo of the affected skin, or consult a veterinarian.")
        elif label == "Health":
            result_type = "healthy"
            message = "No signs of ringworm, flea allergy or scabies were detected."
        elif self.cfg["empty_mask_gate"] and not out["lesion_found"]:
            # the CNN leans towards a disease, but the U-Net found no affected skin
            result_type, reason = "uncertain", "no_lesion_found"
            message = (f"The result is unclear. The model leaned towards {label} ({conf:.0%}), "
                       f"but it could not find an affected skin area in this photo. "
                       f"Try a clear, well-lit photo of the area you are worried about, "
                       f"or consult a veterinarian.")
        else:
            result_type = "disease"
            message = f"Possible {DISEASE_INFO[label]['name']} detected."

        # show the lesion highlight whenever the top guess is a disease
        # (also for 'uncertain'), never when the top guess is Health
        show_lesion = result_type != "healthy" and out["lesion_found"]
        overlay = make_overlay(img, out["mask"], out["box"]) if show_lesion else None

        if result_type == "uncertain":
            # info for the most likely DISEASE ("If it is ...")
            info_key = max((c for c in self.class_names if c != "Health"),
                           key=lambda c: probs[self.class_names.index(c)])
        else:
            info_key = label

        return {
            "status": "ok",
            "result_type": result_type,              # disease | healthy | uncertain
            "uncertain_reason": reason,              # low_confidence | no_lesion_found | None
            "message": message,
            "prediction": {
                "label": label,
                "confidence": round(conf, 4),
                "second_label": second,
                "probabilities": {c: round(float(p), 4)
                                  for c, p in zip(self.class_names, probs)},
            },
            "lesion": {
                "found": out["lesion_found"],
                "shown": show_lesion,
                "area_pct": round(out["lesion_area_pct"], 2),
                "box": list(out["box"]) if out["box"] is not None else None,
            },
            "disease_info": DISEASE_INFO[info_key],
            "images": {
                "original": _jpeg_data_url(img),
                "overlay": _jpeg_data_url(overlay) if overlay is not None else None,
            },
            "image_size": list(img.size),
            "disclaimer": DISCLAIMER,
            "processing_ms": int((time.time() - t0) * 1000),
        }
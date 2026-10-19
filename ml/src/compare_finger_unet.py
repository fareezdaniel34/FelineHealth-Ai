"""
Compare the original U-Net and finger-aware U-Net specifically on
manually annotated finger regions.

The script:

1. Reads finger rectangles from finger_manifest.csv.
2. Uses validation photos by default.
3. Prevents duplicate variants from being counted more than once.
4. Letterboxes each selected photo to 96x96.
5. Transforms the finger rectangles into letterboxed coordinates.
6. Runs both U-Net models.
7. Applies each model's saved post-processing settings.
8. Measures false-positive lesion pixels inside the finger regions.
9. Creates visual comparison images.

Run from the FelineHealth-Ai project root:

    python ml/src/compare_finger_unet.py

Optional test-set evaluation:

    python ml/src/compare_finger_unet.py --split test
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.append(str(Path(__file__).parent))

import unet_np as un
from tune_unet_postprocess import postprocess


# ============================================================
# File helpers
# ============================================================

def read_manifest(path):
    path = Path(path)

    if not path.exists():
        sys.exit(
            f"Manifest not found: {path}\n"
            "Run finger_hard_negative.py scan and annotate first."
        )

    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def load_postprocess_config(path):
    path = Path(path)

    if not path.exists():
        sys.exit(
            f"Post-processing settings not found: {path}"
        )

    with open(
        path,
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    return {
        "threshold": float(config["threshold"]),
        "min_blob_px": int(config["min_blob_px"]),
        "img_size": int(config.get("img_size", 96)),
    }


# ============================================================
# Image preparation
# ============================================================

def prepare_letterboxed_image(image_path, output_size):
    """
    Apply EXIF correction and letterbox the photo into a square.

    Returns:
        model_array:
            RGB uint8 array with shape (output_size, output_size, 3)

        transform:
            Information needed to transform original finger rectangles
            into the letterboxed model coordinates.

        original_image:
            Correctly oriented PIL RGB image.
    """

    with Image.open(image_path) as opened_image:
        original_image = ImageOps.exif_transpose(
            opened_image
        ).convert("RGB")

    original_width, original_height = original_image.size
    square_side = max(original_width, original_height)

    offset_x = (square_side - original_width) // 2
    offset_y = (square_side - original_height) // 2

    canvas = Image.new(
        "RGB",
        (square_side, square_side),
        color=(0, 0, 0),
    )

    canvas.paste(
        original_image,
        (offset_x, offset_y),
    )

    resized = canvas.resize(
        (output_size, output_size),
        Image.Resampling.BILINEAR,
    )

    transform = {
        "original_width": original_width,
        "original_height": original_height,
        "square_side": square_side,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "scale": output_size / square_side,
    }

    return (
        np.asarray(resized, dtype=np.uint8),
        transform,
        original_image,
    )


def transform_box(box, transform, output_size):
    """
    Transform an original-image rectangle into letterboxed model
    coordinates.
    """

    x0, y0, x1, y1 = box

    scale = transform["scale"]
    offset_x = transform["offset_x"]
    offset_y = transform["offset_y"]

    transformed_x0 = int(
        np.floor((x0 + offset_x) * scale)
    )

    transformed_y0 = int(
        np.floor((y0 + offset_y) * scale)
    )

    transformed_x1 = int(
        np.ceil((x1 + offset_x) * scale)
    )

    transformed_y1 = int(
        np.ceil((y1 + offset_y) * scale)
    )

    transformed_x0 = int(
        np.clip(transformed_x0, 0, output_size - 1)
    )

    transformed_y0 = int(
        np.clip(transformed_y0, 0, output_size - 1)
    )

    transformed_x1 = int(
        np.clip(transformed_x1, transformed_x0 + 1, output_size)
    )

    transformed_y1 = int(
        np.clip(transformed_y1, transformed_y0 + 1, output_size)
    )

    return [
        transformed_x0,
        transformed_y0,
        transformed_x1,
        transformed_y1,
    ]


def make_finger_region(boxes, transform, output_size):
    """
    Create one boolean mask representing all annotated finger rectangles.
    """

    finger_region = np.zeros(
        (output_size, output_size),
        dtype=bool,
    )

    transformed_boxes = []

    for box in boxes:
        transformed_box = transform_box(
            box,
            transform,
            output_size,
        )

        x0, y0, x1, y1 = transformed_box

        finger_region[y0:y1, x0:x1] = True
        transformed_boxes.append(transformed_box)

    return finger_region, transformed_boxes


# ============================================================
# Model prediction
# ============================================================

def load_model(model_path, base):
    model_path = Path(model_path)

    if not model_path.exists():
        sys.exit(
            f"Model not found: {model_path}"
        )

    model = un.UNet(
        base=base,
        seed=42,
    )

    model.load(str(model_path))

    return model


def predict_one(model, image_array, config):
    """
    Predict one post-processed binary mask.
    """

    model_input = (
        image_array.astype(np.float32) / 255.0
    )[None, ...]

    probability = model.predict_mask_proba(
        model_input,
        batch_size=1,
    )[0]

    prediction = postprocess(
        probability,
        config["threshold"],
        config["min_blob_px"],
    )

    return probability, prediction


# ============================================================
# Metrics
# ============================================================

def calculate_finger_metrics(prediction, probability, finger_region):
    finger_pixels = int(finger_region.sum())

    if finger_pixels == 0:
        return {
            "finger_pixels": 0,
            "false_positive_pixels": 0,
            "false_positive_percent": 0.0,
            "mean_finger_probability": 0.0,
            "max_finger_probability": 0.0,
            "finger_detected": False,
        }

    predicted_inside_finger = (
        prediction & finger_region
    )

    false_positive_pixels = int(
        predicted_inside_finger.sum()
    )

    finger_probabilities = probability[
        finger_region
    ]

    return {
        "finger_pixels": finger_pixels,
        "false_positive_pixels": false_positive_pixels,
        "false_positive_percent": (
            100.0
            * false_positive_pixels
            / finger_pixels
        ),
        "mean_finger_probability": float(
            finger_probabilities.mean()
        ),
        "max_finger_probability": float(
            finger_probabilities.max()
        ),
        "finger_detected": bool(
            false_positive_pixels > 0
        ),
    }


# ============================================================
# Visualisation
# ============================================================

def create_overlay(
    image_array,
    prediction,
    finger_region,
    transformed_boxes,
    title,
):
    """
    Red indicates predicted lesion.
    Green rectangles indicate annotated finger regions.
    Yellow indicates prediction inside a finger rectangle.
    """

    base_image = Image.fromarray(
        image_array
    ).convert("RGBA")

    overlay = np.zeros(
        (
            image_array.shape[0],
            image_array.shape[1],
            4,
        ),
        dtype=np.uint8,
    )

    # Red for predicted lesion.
    overlay[prediction] = [
        255,
        0,
        0,
        125,
    ]

    # Yellow for predicted lesion inside finger area.
    predicted_finger = (
        prediction & finger_region
    )

    overlay[predicted_finger] = [
        255,
        255,
        0,
        210,
    ]

    overlay_image = Image.fromarray(
        overlay,
        mode="RGBA",
    )

    combined = Image.alpha_composite(
        base_image,
        overlay_image,
    )

    draw = ImageDraw.Draw(combined)

    for box in transformed_boxes:
        x0, y0, x1, y1 = box

        draw.rectangle(
            [x0, y0, x1 - 1, y1 - 1],
            outline=(0, 255, 0, 255),
            width=2,
        )

    # Enlarge for easier inspection.
    combined = combined.resize(
        (288, 288),
        Image.Resampling.NEAREST,
    )

    title_area = Image.new(
        "RGBA",
        (288, 324),
        color=(20, 20, 20, 255),
    )

    title_area.paste(
        combined,
        (0, 0),
    )

    title_draw = ImageDraw.Draw(
        title_area
    )

    title_draw.text(
        (5, 294),
        title[:44],
        fill=(255, 255, 255, 255),
    )

    return title_area.convert("RGB")


def create_comparison_sheet(preview_rows, output_path):
    if not preview_rows:
        return

    column_width = 288
    row_height = 324

    sheet = Image.new(
        "RGB",
        (
            column_width * 2,
            row_height * len(preview_rows),
        ),
        color=(0, 0, 0),
    )

    for row_index, preview in enumerate(
        preview_rows
    ):
        sheet.paste(
            preview["old"],
            (0, row_index * row_height),
        )

        sheet.paste(
            preview["new"],
            (
                column_width,
                row_index * row_height,
            ),
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(output_path)


# ============================================================
# Comparison
# ============================================================

def compare_models(args):
    manifest_rows = read_manifest(
        args.manifest
    )

    old_config = load_postprocess_config(
        args.old_postprocess
    )

    new_config = load_postprocess_config(
        args.new_postprocess
    )

    if old_config["img_size"] != new_config["img_size"]:
        sys.exit(
            "Original and candidate models use different image sizes."
        )

    image_size = old_config["img_size"]

    print("Loading original U-Net...")

    old_model = load_model(
        args.old_model,
        args.base,
    )

    print("Loading finger-aware U-Net...")

    new_model = load_model(
        args.new_model,
        args.base,
    )

    # Select only approved, annotated photos belonging to the requested
    # clean-data split.
    eligible_rows = [
        row
        for row in manifest_rows
        if row.get("approved") == "True"
        and row.get("finger_boxes")
        and row.get("matched_split") == args.split
    ]

    if not eligible_rows:
        sys.exit(
            f"No approved finger photos were found for split: "
            f"{args.split}"
        )

    # Do not count multiple transformed/exported versions of the same
    # original source image more than once.
    selected_rows = []
    used_groups = set()

    for row in eligible_rows:
        duplicate_group = row["duplicate_group"]

        if duplicate_group in used_groups:
            continue

        used_groups.add(duplicate_group)
        selected_rows.append(row)

    print()
    print(
        f"Split: {args.split}"
    )

    print(
        f"Annotated files: {len(eligible_rows)}"
    )

    print(
        f"Unique source groups: {len(selected_rows)}"
    )

    if (
        args.split == "val"
        and len(selected_rows) < 3
    ):
        print(
            "WARNING: Fewer than 3 unique validation finger sources."
        )

        print(
            "The result is useful as a targeted check, but is not "
            "strong evidence of generalisation."
        )

    result_rows = []
    preview_rows = []

    total_finger_pixels = 0
    total_old_fp_pixels = 0
    total_new_fp_pixels = 0

    old_detected_count = 0
    new_detected_count = 0

    for index, row in enumerate(
        selected_rows,
        start=1,
    ):
        selected_path = Path(
            row["selected_path"]
        )

        if not selected_path.exists():
            print(
                f"Skipping missing file: {selected_path}"
            )
            continue

        boxes = json.loads(
            row["finger_boxes"]
        )

        (
            image_array,
            transform,
            _,
        ) = prepare_letterboxed_image(
            selected_path,
            image_size,
        )

        (
            finger_region,
            transformed_boxes,
        ) = make_finger_region(
            boxes,
            transform,
            image_size,
        )

        old_probability, old_prediction = predict_one(
            old_model,
            image_array,
            old_config,
        )

        new_probability, new_prediction = predict_one(
            new_model,
            image_array,
            new_config,
        )

        old_metrics = calculate_finger_metrics(
            old_prediction,
            old_probability,
            finger_region,
        )

        new_metrics = calculate_finger_metrics(
            new_prediction,
            new_probability,
            finger_region,
        )

        total_finger_pixels += (
            old_metrics["finger_pixels"]
        )

        total_old_fp_pixels += (
            old_metrics["false_positive_pixels"]
        )

        total_new_fp_pixels += (
            new_metrics["false_positive_pixels"]
        )

        old_detected_count += int(
            old_metrics["finger_detected"]
        )

        new_detected_count += int(
            new_metrics["finger_detected"]
        )

        if (
            old_metrics["false_positive_pixels"] > 0
        ):
            image_reduction = (
                100.0
                * (
                    old_metrics[
                        "false_positive_pixels"
                    ]
                    - new_metrics[
                        "false_positive_pixels"
                    ]
                )
                / old_metrics[
                    "false_positive_pixels"
                ]
            )
        elif (
            new_metrics["false_positive_pixels"] == 0
        ):
            image_reduction = 0.0
        else:
            image_reduction = float("-inf")

        result_row = {
            "file": selected_path.name,
            "split": args.split,
            "duplicate_group": row[
                "duplicate_group"
            ],
            "finger_pixels": old_metrics[
                "finger_pixels"
            ],
            "old_fp_pixels": old_metrics[
                "false_positive_pixels"
            ],
            "new_fp_pixels": new_metrics[
                "false_positive_pixels"
            ],
            "old_fp_percent": old_metrics[
                "false_positive_percent"
            ],
            "new_fp_percent": new_metrics[
                "false_positive_percent"
            ],
            "old_mean_probability": old_metrics[
                "mean_finger_probability"
            ],
            "new_mean_probability": new_metrics[
                "mean_finger_probability"
            ],
            "old_finger_detected": old_metrics[
                "finger_detected"
            ],
            "new_finger_detected": new_metrics[
                "finger_detected"
            ],
            "fp_reduction_percent": image_reduction,
        }

        result_rows.append(
            result_row
        )

        old_title = (
            f"OLD: {selected_path.name} | "
            f"FP {old_metrics['false_positive_pixels']}px"
        )

        new_title = (
            f"NEW: {selected_path.name} | "
            f"FP {new_metrics['false_positive_pixels']}px"
        )

        preview_rows.append({
            "old": create_overlay(
                image_array,
                old_prediction,
                finger_region,
                transformed_boxes,
                old_title,
            ),
            "new": create_overlay(
                image_array,
                new_prediction,
                finger_region,
                transformed_boxes,
                new_title,
            ),
        })

        print(
            f"\n[{index}/{len(selected_rows)}] "
            f"{selected_path.name}"
        )

        print(
            "  Original: "
            f"{old_metrics['false_positive_pixels']} px "
            f"({old_metrics['false_positive_percent']:.1f}% "
            "of finger region)"
        )

        print(
            "  Candidate: "
            f"{new_metrics['false_positive_pixels']} px "
            f"({new_metrics['false_positive_percent']:.1f}% "
            "of finger region)"
        )

    if not result_rows:
        sys.exit(
            "No photos could be evaluated."
        )

    if total_old_fp_pixels > 0:
        overall_reduction = (
            100.0
            * (
                total_old_fp_pixels
                - total_new_fp_pixels
            )
            / total_old_fp_pixels
        )
    elif total_new_fp_pixels == 0:
        overall_reduction = 0.0
    else:
        overall_reduction = float("-inf")

    old_fp_percent = (
        100.0
        * total_old_fp_pixels
        / max(total_finger_pixels, 1)
    )

    new_fp_percent = (
        100.0
        * total_new_fp_pixels
        / max(total_finger_pixels, 1)
    )

    output_folder = Path(
        args.output
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_folder
        / f"finger_comparison_{args.split}.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                result_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(result_rows)

    preview_path = (
        output_folder
        / f"finger_comparison_{args.split}.png"
    )

    create_comparison_sheet(
        preview_rows,
        preview_path,
    )

    summary = {
        "split": args.split,
        "unique_source_groups": len(result_rows),
        "total_finger_pixels": total_finger_pixels,
        "old_false_positive_pixels": (
            total_old_fp_pixels
        ),
        "new_false_positive_pixels": (
            total_new_fp_pixels
        ),
        "old_false_positive_percent": (
            old_fp_percent
        ),
        "new_false_positive_percent": (
            new_fp_percent
        ),
        "false_positive_reduction_percent": (
            overall_reduction
        ),
        "old_images_with_finger_detection": (
            old_detected_count
        ),
        "new_images_with_finger_detection": (
            new_detected_count
        ),
        "required_reduction_percent": 50.0,
        "passes_reduction_rule": bool(
            total_old_fp_pixels > 0
            and overall_reduction >= 50.0
        ),
    }

    summary_path = (
        output_folder
        / f"finger_comparison_{args.split}.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 60)
    print("FINGER FALSE-POSITIVE SUMMARY")
    print("=" * 60)

    print(
        f"Unique photos evaluated : {len(result_rows)}"
    )

    print(
        f"Original finger FP      : "
        f"{total_old_fp_pixels} / {total_finger_pixels} px "
        f"({old_fp_percent:.2f}%)"
    )

    print(
        f"Candidate finger FP     : "
        f"{total_new_fp_pixels} / {total_finger_pixels} px "
        f"({new_fp_percent:.2f}%)"
    )

    print(
        f"Original images flagged : "
        f"{old_detected_count}/{len(result_rows)}"
    )

    print(
        f"Candidate images flagged: "
        f"{new_detected_count}/{len(result_rows)}"
    )

    if total_old_fp_pixels == 0:
        print(
            "Reduction               : Cannot measure because "
            "the original model produced zero finger false positives "
            "on this split."
        )

        decision = (
            "INCONCLUSIVE: the selected split does not reproduce "
            "the original finger error."
        )

    else:
        print(
            f"Reduction               : "
            f"{overall_reduction:.2f}%"
        )

        if overall_reduction >= 50.0:
            decision = (
                "CANDIDATE PASSES the finger false-positive rule."
            )
        else:
            decision = (
                "CANDIDATE FAILS the required 50% reduction."
            )

    print()
    print(f"DECISION: {decision}")

    if (
        args.split == "val"
        and len(result_rows) < 3
    ):
        print(
            "CONFIDENCE: LOW, because fewer than 3 unique validation "
            "finger sources were available."
        )

    print()
    print(f"CSV report: {csv_path}")
    print(f"JSON report: {summary_path}")
    print(f"Visual comparison: {preview_path}")

    print()
    print(
        "Overlay colours: red = predicted lesion, "
        "green box = finger region, "
        "yellow = predicted lesion inside finger."
    )


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare original and finger-aware U-Net predictions "
            "inside annotated finger regions."
        )
    )

    parser.add_argument(
        "--manifest",
        default=(
            "ml/data/finger_fix/"
            "finger_manifest.csv"
        ),
    )

    parser.add_argument(
        "--old_model",
        default="ml/models/unet_v1.npz",
    )

    parser.add_argument(
        "--new_model",
        default=(
            "ml/models/unet_finger_v1.npz"
        ),
    )

    parser.add_argument(
        "--old_postprocess",
        default=(
            "ml/models/"
            "unet_postprocess_original.json"
        ),
    )

    parser.add_argument(
        "--new_postprocess",
        default=(
            "ml/models/"
            "unet_postprocess_finger.json"
        ),
    )

    parser.add_argument(
        "--base",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "test",
        ],
        default="val",
    )

    parser.add_argument(
        "--output",
        default="ml/results/finger_comparison",
    )

    args = parser.parse_args()
    compare_models(args)


if __name__ == "__main__":
    main()
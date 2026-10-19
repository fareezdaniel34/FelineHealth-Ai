"""
Prepare finger hard-negative samples for the NumPy U-Net.

Run from the FelineHealth-Ai project root:

1. Match selected finger photos to the cleaned dataset:
   python ml/src/finger_hard_negative.py scan \
     --photos "C:/Users/Asus/Desktop/finger_photos"

2. Draw rectangles around fingers:
   python ml/src/finger_hard_negative.py annotate

3. Build the candidate U-Net dataset:
   python ml/src/finger_hard_negative.py build --repeat 3

Only photos matched to ml/data/clean/train are added to training.
Validation and test photos are recorded for evaluation but never added
to the training arrays.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DEFAULT_MANIFEST = Path(
    "ml/data/finger_fix/finger_manifest.csv"
)


# ============================================================
# General helpers
# ============================================================

def get_image_files(folder):
    folder = Path(folder)

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
    )


def calculate_md5(path):
    hasher = hashlib.md5()

    with open(path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(block)

    return hasher.hexdigest()


def canonical_name(path):
    """
    Convert original and Roboflow-exported filenames into one
    comparable source name.

    Examples:
        jamur15_jpeg.jpg
        jamur15_jpeg_jpg.rf.123abc.jpg

    Both become:
        jamur15
    """

    name = Path(path).name

    # Remove Roboflow's generated suffix.
    name = re.sub(
        r"\.rf\.[0-9a-f]+(?=\.[^.]+$)",
        "",
        name,
        flags=re.IGNORECASE,
    )

    stem = Path(name).stem

    # Some Roboflow filenames contain multiple encoded extensions,
    # such as "_jpeg_jpg". Remove all of them, not just the final one.
    previous_stem = None

    while stem != previous_stem:
        previous_stem = stem

        stem = re.sub(
            r"[_\-\s]+(jpg|jpeg|png|webp)$",
            "",
            stem,
            flags=re.IGNORECASE,
        )

    return re.sub(
        r"[^a-z0-9]+",
        "",
        stem.lower(),
    )


def create_visual_hash(path, size=16):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")
        image = image.resize(
            (size, size),
            Image.Resampling.BILINEAR,
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        )

    return (array > array.mean()).reshape(-1)


def visual_distance(hash_a, hash_b):
    return int(
        np.count_nonzero(hash_a != hash_b)
    )


def load_unet_selection(path):
    """
    Load unet_selection.csv if available.

    This is only additional information.
    The clean dataset split is treated as authoritative.
    """

    results = {}

    if not path:
        return results

    path = Path(path)

    if not path.exists():
        return results

    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as file:
        for row in csv.DictReader(file):

            original_path = row.get(
                "orig_path",
                "",
            )

            if not original_path:
                continue

            key = canonical_name(original_path)

            results.setdefault(
                key,
                [],
            ).append(row)

    return results


def write_manifest(path, rows):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "selected_path",
        "selected_class",
        "canonical",
        "duplicate_group",
        "matched_path",
        "matched_split",
        "matched_class",
        "match_method",
        "match_distance",
        "split_conflict",
        "unet_ids",
        "unet_splits",
        "eligible_train",
        "finger_boxes",
        "approved",
        "notes",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    path = Path(path)

    if not path.exists():
        sys.exit(
            f"Manifest not found: {path}\n"
            "Run the scan command first."
        )

    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


# ============================================================
# Stage 1: Match selected photos to the clean dataset
# ============================================================

def scan_photos(args):
    selected_photos = get_image_files(
        args.photos
    )

    clean_photos = get_image_files(
        args.clean
    )

    if not selected_photos:
        sys.exit(
            f"No images found under:\n{args.photos}"
        )

    if not clean_photos:
        sys.exit(
            f"No cleaned dataset images found under:\n"
            f"{args.clean}"
        )

    print(
        f"Selected photos: {len(selected_photos)}"
    )

    print(
        f"Clean dataset images: {len(clean_photos)}"
    )

    # Build clean dataset indexes.
    exact_index = {}
    name_index = {}
    clean_visual_hashes = {}

    print("\nIndexing clean dataset...")

    for index, path in enumerate(
        clean_photos,
        start=1,
    ):
        exact_index.setdefault(
            calculate_md5(path),
            [],
        ).append(path)

        name_index.setdefault(
            canonical_name(path),
            [],
        ).append(path)

        clean_visual_hashes[path] = (
            create_visual_hash(path)
        )

        if index % 100 == 0:
            print(
                f"  indexed {index}/"
                f"{len(clean_photos)}"
            )

    unet_selection = load_unet_selection(
        args.selection
    )

    selected_visual_hashes = {
        path: create_visual_hash(path)
        for path in selected_photos
    }

    # Group duplicate filenames or Roboflow variants.
    duplicate_groups = {}

    for path in selected_photos:
        key = canonical_name(path)

        duplicate_groups.setdefault(
            key,
            [],
        ).append(path)

    group_ids = {}

    for group_number, paths in enumerate(
        duplicate_groups.values(),
        start=1,
    ):
        group_name = f"group_{group_number:02d}"

        for path in paths:
            group_ids[path] = group_name

    rows = []

    print("\nMatching selected photos...")

    for selected_path in selected_photos:
        selected_md5 = calculate_md5(
            selected_path
        )

        exact_matches = exact_index.get(
            selected_md5,
            [],
        )

        ranked_matches = []
        match_method = ""
        best_distance = ""

        if exact_matches:
            ranked_matches = [
                (0, path)
                for path in exact_matches
            ]

            match_method = "identical"

        else:
            name_matches = name_index.get(
                canonical_name(selected_path),
                [],
            )

            if name_matches:
                candidate_pool = name_matches
                allowed_distance = (
                    args.name_distance
                )
                match_method = "name+visual"

            else:
                candidate_pool = clean_photos
                allowed_distance = (
                    args.global_distance
                )
                match_method = "near-identical"

            all_distances = sorted(
                (
                    visual_distance(
                        selected_visual_hashes[
                            selected_path
                        ],
                        clean_visual_hashes[
                            clean_path
                        ],
                    ),
                    clean_path,
                )
                for clean_path in candidate_pool
            )

            if all_distances:
                best_distance = str(
                    all_distances[0][0]
                )

            ranked_matches = [
                (distance, path)
                for distance, path
                in all_distances
                if distance <= allowed_distance
            ]

        if ranked_matches:
            matched_path = ranked_matches[0][1]

            relative_parts = (
                matched_path.relative_to(
                    Path(args.clean)
                ).parts
            )

            matched_split = (
                relative_parts[0]
                if len(relative_parts) >= 1
                else ""
            )

            matched_class = (
                relative_parts[1]
                if len(relative_parts) >= 2
                else ""
            )

            candidate_splits = sorted({
                path.relative_to(
                    Path(args.clean)
                ).parts[0]
                for _, path in ranked_matches
            })

            split_conflict = (
                len(candidate_splits) > 1
            )

        else:
            matched_path = None
            matched_split = ""
            matched_class = ""
            split_conflict = False
            match_method = "not_found"

        selection_rows = (
            unet_selection.get(
                canonical_name(selected_path),
                [],
            )
        )

        unet_ids = ";".join(
            row.get("unet_id", "")
            for row in selection_rows
        )

        unet_splits = ";".join(
            sorted({
                row.get("unet_split", "")
                for row in selection_rows
                if row.get("unet_split", "")
            })
        )

        eligible_train = bool(
            matched_path
            and matched_split == "train"
            and not split_conflict
        )

        row = {
            "selected_path": str(
                selected_path.resolve()
            ),
            "selected_class": (
                selected_path.parent.name
            ),
            "canonical": canonical_name(
                selected_path
            ),
            "duplicate_group": (
                group_ids[selected_path]
            ),
            "matched_path": (
                str(matched_path.resolve())
                if matched_path
                else ""
            ),
            "matched_split": matched_split,
            "matched_class": matched_class,
            "match_method": match_method,
            "match_distance": best_distance,
            "split_conflict": str(
                split_conflict
            ),
            "unet_ids": unet_ids,
            "unet_splits": unet_splits,
            "eligible_train": str(
                eligible_train
            ),
            "finger_boxes": "",
            "approved": "",
            "notes": "",
        }

        rows.append(row)

        print(
            f"{selected_path.name[:45]:45} "
            f"split={matched_split or 'NONE':5} "
            f"method={match_method:14} "
            f"train={str(eligible_train):5} "
            f"group={group_ids[selected_path]}"
        )

    write_manifest(
        args.manifest,
        rows,
    )

    eligible_count = sum(
        row["eligible_train"] == "True"
        for row in rows
    )

    matched_count = sum(
        bool(row["matched_path"])
        for row in rows
    )

    print("\nScan complete.")

    print(
        f"Matched: {matched_count}/"
        f"{len(rows)}"
    )

    print(
        "Eligible training photos before "
        f"duplicate removal: {eligible_count}/"
        f"{len(rows)}"
    )

    print(
        f"Manifest saved to:\n{args.manifest}"
    )

    print(
        "\nNext command:\n"
        "python ml/src/"
        "finger_hard_negative.py annotate"
    )


# ============================================================
# Stage 2: Draw rectangles around fingers
# ============================================================

def annotate_fingers(args):
    try:
        import matplotlib.pyplot as plt

        from matplotlib.patches import (
            Rectangle,
        )

        from matplotlib.widgets import (
            RectangleSelector,
        )

    except ImportError:
        sys.exit(
            "matplotlib is required.\n"
            "Install or run using the Python "
            "environment used for your project."
        )

    rows = read_manifest(
        args.manifest
    )

    changed = False

    for row_index, row in enumerate(
        rows,
        start=1,
    ):
        if not row["matched_path"]:
            continue

        if (
            row["finger_boxes"]
            and not args.redo
        ):
            continue

        selected_path = Path(
            row["selected_path"]
        )

        with Image.open(
            selected_path
        ) as opened_image:
            image = ImageOps.exif_transpose(
                opened_image
            ).convert("RGB")

        boxes = []
        state = {
            "quit": False,
        }

        figure, axis = plt.subplots(
            figsize=(11, 8)
        )

        axis.imshow(image)

        axis.set_title(
            f"{row_index}/{len(rows)}: "
            f"{selected_path.name}\n"
            f"Clean split: "
            f"{row['matched_split']} | "
            "Draw a tight rectangle around "
            "each FINGER only\n"
            "Enter = save | "
            "Backspace = undo | "
            "S = skip | Q = save and quit"
        )

        axis.axis("off")

        def rectangle_selected(
            click_event,
            release_event,
        ):
            coordinates = (
                click_event.xdata,
                click_event.ydata,
                release_event.xdata,
                release_event.ydata,
            )

            if any(
                value is None
                for value in coordinates
            ):
                return

            x0, x1 = sorted([
                int(round(click_event.xdata)),
                int(round(release_event.xdata)),
            ])

            y0, y1 = sorted([
                int(round(click_event.ydata)),
                int(round(release_event.ydata)),
            ])

            x0 = max(0, x0)
            y0 = max(0, y0)

            x1 = min(image.width, x1)
            y1 = min(image.height, y1)

            if (
                x1 - x0 < 5
                or y1 - y0 < 5
            ):
                return

            boxes.append([
                x0,
                y0,
                x1,
                y1,
            ])

            rectangle = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="lime",
                linewidth=2,
            )

            axis.add_patch(rectangle)
            figure.canvas.draw_idle()

        selector = RectangleSelector(
            axis,
            rectangle_selected,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )

        def key_pressed(event):
            nonlocal changed

            if (
                event.key == "backspace"
                and boxes
            ):
                boxes.pop()

                if axis.patches:
                    axis.patches[-1].remove()

                figure.canvas.draw_idle()

            elif event.key == "enter":
                if not boxes:
                    print(
                        "Draw at least one finger "
                        "rectangle, or press S."
                    )
                    return

                row["finger_boxes"] = json.dumps(
                    boxes,
                    separators=(",", ":"),
                )

                row["approved"] = "True"
                row["notes"] = ""

                changed = True

                plt.close(figure)

            elif (
                event.key
                and event.key.lower() == "s"
            ):
                row["approved"] = "False"

                row["notes"] = (
                    "Skipped during annotation"
                )

                changed = True

                plt.close(figure)

            elif (
                event.key
                and event.key.lower() == "q"
            ):
                state["quit"] = True
                plt.close(figure)

        figure.canvas.mpl_connect(
            "key_press_event",
            key_pressed,
        )

        plt.show()

        selector.set_active(False)

        if changed:
            write_manifest(
                args.manifest,
                rows,
            )

        if state["quit"]:
            break

    write_manifest(
        args.manifest,
        rows,
    )

    print(
        "\nFinger annotations saved to:"
    )

    print(args.manifest)

    print(
        "\nNext command:\n"
        "python ml/src/"
        "finger_hard_negative.py build "
        "--repeat 3"
    )


# ============================================================
# Stage 3: Create hard-negative crops
# ============================================================

def crop_with_margin(
    image,
    box,
    margin,
):
    x0, y0, x1, y1 = box

    box_width = x1 - x0
    box_height = y1 - y0

    margin_x = int(
        round(box_width * margin)
    )

    margin_y = int(
        round(box_height * margin)
    )

    crop_box = (
        max(0, x0 - margin_x),
        max(0, y0 - margin_y),
        min(image.width, x1 + margin_x),
        min(image.height, y1 + margin_y),
    )

    return image.crop(crop_box)


def letterbox_image(
    image,
    output_size,
):
    width, height = image.size
    side = max(width, height)

    canvas = Image.new(
        "RGB",
        (side, side),
        color=(0, 0, 0),
    )

    offset_x = (side - width) // 2
    offset_y = (side - height) // 2

    canvas.paste(
        image,
        (offset_x, offset_y),
    )

    return canvas.resize(
        (output_size, output_size),
        Image.Resampling.BILINEAR,
    )


def build_hard_negative_dataset(args):
    rows = read_manifest(
        args.manifest
    )

    source_folder = Path(
        args.data
    )

    output_folder = Path(
        args.output
    )

    required_files = []

    for split in (
        "train",
        "val",
        "test",
    ):
        required_files.append(
            source_folder / f"X_{split}.npy"
        )

        required_files.append(
            source_folder / f"M_{split}.npy"
        )

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        sys.exit(
            "Missing original U-Net arrays:\n"
            + "\n".join(missing_files)
        )

    train_images = np.load(
        source_folder / "X_train.npy"
    )

    train_masks = np.load(
        source_folder / "M_train.npy"
    )

    if train_images.shape[1] != args.size:
        sys.exit(
            "Original U-Net images use "
            f"{train_images.shape[1]}px, "
            f"but --size is {args.size}."
        )

    additions = []
    addition_ids = []
    preview_items = []

    used_duplicate_groups = set()
    rejected_non_train = []

    for row in rows:
        if row["approved"] != "True":
            continue

        if not row["finger_boxes"]:
            continue

        # Never allow validation or test images
        # into training.
        if (
            row["eligible_train"] != "True"
            or row["matched_split"] != "train"
        ):
            rejected_non_train.append({
                "file": Path(
                    row["selected_path"]
                ).name,
                "split": (
                    row["matched_split"]
                    or "unmatched"
                ),
            })

            continue

        duplicate_group = (
            row["duplicate_group"]
        )

        # Use only one representative from
        # duplicate variants.
        if (
            duplicate_group
            in used_duplicate_groups
        ):
            continue

        used_duplicate_groups.add(
            duplicate_group
        )

        # The rectangles were drawn on
        # selected_path, so use that exact image.
        # matched_path is only used to confirm
        # that the photo belongs to the clean
        # training split.
        selected_path = Path(
            row["selected_path"]
        )

        with Image.open(
            selected_path
        ) as opened_image:
            image = ImageOps.exif_transpose(
                opened_image
            ).convert("RGB")

        finger_boxes = json.loads(
            row["finger_boxes"]
        )

        for box_index, box in enumerate(
            finger_boxes,
            start=1,
        ):
            finger_crop = crop_with_margin(
                image,
                box,
                args.margin,
            )

            resized_crop = letterbox_image(
                finger_crop,
                args.size,
            )

            crop_array = np.asarray(
                resized_crop,
                dtype=np.uint8,
            )

            for repeat_index in range(
                args.repeat
            ):
                additions.append(
                    crop_array.copy()
                )

                addition_ids.append(
                    f"finger_negative_"
                    f"{duplicate_group}_"
                    f"{box_index}_"
                    f"repeat_"
                    f"{repeat_index + 1}"
                )

            preview_items.append({
                "array": crop_array,
                "name": selected_path.name,
            })

    if not additions:
        sys.exit(
            "No eligible training finger crops "
            "were created.\n"
            "Check finger_manifest.csv and make "
            "sure some approved photos belong "
            "to the clean TRAIN split."
        )

    added_images = np.stack(
        additions,
        axis=0,
    )

    if train_masks.ndim == 4:
        mask_shape = (
            len(added_images),
            args.size,
            args.size,
            train_masks.shape[-1],
        )
    else:
        mask_shape = (
            len(added_images),
            args.size,
            args.size,
        )

    # Every finger hard-negative crop gets an
    # empty mask because the finger is background.
    added_masks = np.zeros(
        mask_shape,
        dtype=train_masks.dtype,
    )

    combined_images = np.concatenate(
        [
            train_images,
            added_images,
        ],
        axis=0,
    )

    combined_masks = np.concatenate(
        [
            train_masks,
            added_masks,
        ],
        axis=0,
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_folder / "X_train.npy",
        combined_images,
    )

    np.save(
        output_folder / "M_train.npy",
        combined_masks,
    )

    # Validation and test sets remain unchanged.
    for split in (
        "val",
        "test",
    ):
        shutil.copy2(
            source_folder / f"X_{split}.npy",
            output_folder / f"X_{split}.npy",
        )

        shutil.copy2(
            source_folder / f"M_{split}.npy",
            output_folder / f"M_{split}.npy",
        )

    # Preserve dataset IDs when available.
    for split in (
        "train",
        "val",
        "test",
    ):
        source_ids = (
            source_folder
            / f"ids_{split}.json"
        )

        if not source_ids.exists():
            continue

        with open(
            source_ids,
            encoding="utf-8",
        ) as file:
            ids = json.load(file)

        if split == "train":
            ids.extend(addition_ids)

        with open(
            output_folder
            / f"ids_{split}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                ids,
                file,
                indent=2,
            )

    create_preview(
        preview_items
    )

    results_folder = Path(
        "ml/results"
    )

    results_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "source_dataset": str(
            source_folder
        ),
        "output_dataset": str(
            output_folder
        ),
        "original_training_images": int(
            len(train_images)
        ),
        "added_training_arrays": int(
            len(added_images)
        ),
        "unique_finger_crops": int(
            len(preview_items)
        ),
        "repeat_factor": int(
            args.repeat
        ),
        "rejected_non_train": (
            rejected_non_train
        ),
    }

    with open(
        results_folder
        / "finger_fix_build.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\nCandidate dataset created.")

    print(
        f"Original training samples: "
        f"{len(train_images)}"
    )

    print(
        f"Unique finger crops: "
        f"{len(preview_items)}"
    )

    print(
        f"Added arrays after repeat: "
        f"{len(added_images)}"
    )

    print(
        f"New total training samples: "
        f"{len(combined_images)}"
    )

    if rejected_non_train:
        print(
            "Validation/test/unmatched photos "
            f"safely rejected: "
            f"{len(rejected_non_train)}"
        )

    print(
        "\nDataset saved to:"
    )

    print(output_folder)

    print(
        "\nIMPORTANT: Open and inspect:"
    )

    print(
        "ml/results/"
        "finger_negative_crops.png"
    )

    print(
        "\nEvery preview tile must contain "
        "only a finger and safe background."
    )

    print(
        "If a crop includes the real cat "
        "lesion, do not train yet."
    )


def create_preview(preview_items):
    if not preview_items:
        return

    result_folder = Path(
        "ml/results"
    )

    result_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    tiles = []

    for item in preview_items:
        image = Image.fromarray(
            item["array"]
        )

        image = image.resize(
            (192, 192),
            Image.Resampling.NEAREST,
        )

        draw = ImageDraw.Draw(image)

        draw.rectangle(
            (0, 164, 191, 191),
            fill=(0, 0, 0),
        )

        draw.text(
            (4, 169),
            item["name"][:29],
            fill=(255, 255, 255),
        )

        tiles.append(
            np.asarray(image)
        )

    column_count = min(
        4,
        len(tiles),
    )

    image_rows = []

    for start_index in range(
        0,
        len(tiles),
        column_count,
    ):
        current_row = tiles[
            start_index:
            start_index + column_count
        ]

        while (
            len(current_row)
            < column_count
        ):
            current_row.append(
                np.zeros_like(tiles[0])
            )

        image_rows.append(
            np.concatenate(
                current_row,
                axis=1,
            )
        )

    preview = np.concatenate(
        image_rows,
        axis=0,
    )

    Image.fromarray(
        preview
    ).save(
        result_folder
        / "finger_negative_crops.png"
    )


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare split-safe finger "
            "hard negatives for U-Net."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    scan_parser = subparsers.add_parser(
        "scan"
    )

    scan_parser.add_argument(
        "--photos",
        required=True,
        help=(
            "Folder containing the selected "
            "finger photos."
        ),
    )

    scan_parser.add_argument(
        "--clean",
        default="ml/data/clean",
    )

    scan_parser.add_argument(
        "--selection",
        default=(
            "ml/data/unet_selection.csv"
        ),
    )

    scan_parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
    )

    scan_parser.add_argument(
        "--name_distance",
        type=int,
        default=80,
    )

    scan_parser.add_argument(
        "--global_distance",
        type=int,
        default=10,
    )

    scan_parser.set_defaults(
        function=scan_photos
    )

    annotate_parser = (
        subparsers.add_parser(
            "annotate"
        )
    )

    annotate_parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
    )

    annotate_parser.add_argument(
        "--redo",
        action="store_true",
    )

    annotate_parser.set_defaults(
        function=annotate_fingers
    )

    build_parser = subparsers.add_parser(
        "build"
    )

    build_parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
    )

    build_parser.add_argument(
        "--data",
        default="ml/data/unet_96",
    )

    build_parser.add_argument(
        "--output",
        default=(
            "ml/data/unet_96_finger"
        ),
    )

    build_parser.add_argument(
        "--size",
        type=int,
        default=96,
    )

    build_parser.add_argument(
        "--repeat",
        type=int,
        default=3,
    )

    build_parser.add_argument(
        "--margin",
        type=float,
        default=0.10,
    )

    build_parser.set_defaults(
        function=build_hard_negative_dataset
    )

    args = parser.parse_args()

    args.manifest = Path(
        args.manifest
    )

    args.function(args)


if __name__ == "__main__":
    main()
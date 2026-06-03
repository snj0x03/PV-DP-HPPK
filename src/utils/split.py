# Train/val/test splitting logic for classification-format image datasets.
# Supports two modes:
#   random — shuffle individual files per class (simple baseline)
#   chunk  — shuffle groups of consecutive frames (prevents near-duplicate leakage)

import os
import random
import shutil


def _collect_files(source_dir: str) -> dict:
    """
    Scan source_dir for class subfolders and return their sorted image paths.
    Returns {class_name: [absolute_path, ...]} with files sorted alphabetically.
    """
    files_by_class = {}
    for class_name in sorted(os.listdir(source_dir)):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        files = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if files:
            files_by_class[class_name] = [os.path.join(class_path, f) for f in files]
    return files_by_class


def _cut_at_ratios(items: list, ratios: list) -> tuple:
    """Split a flat list into (train, val, test) at the given ratio boundaries."""
    n         = len(items)
    train_end = int(n * ratios[0])
    val_end   = train_end + int(n * ratios[1])
    return items[:train_end], items[train_end:val_end], items[val_end:]


def random_split(files_by_class: dict, ratios: list, seed: int) -> dict:
    """
    Shuffle each class's file list independently, then split at ratio boundaries.
    Each class is shuffled with the same seed so results are reproducible.

    Returns {"train": {class: [paths]}, "val": ..., "test": ...}
    """
    rng    = random.Random(seed)
    splits = {"train": {}, "val": {}, "test": {}}

    for cls, files in files_by_class.items():
        shuffled = files[:]
        rng.shuffle(shuffled)
        train, val, test = _cut_at_ratios(shuffled, ratios)
        splits["train"][cls] = train
        splits["val"][cls]   = val
        splits["test"][cls]  = test

    return splits


def chunk_split(files_by_class: dict, ratios: list, chunk_size: int, seed: int) -> dict:
    """
    Group consecutive files into chunks of chunk_size, shuffle the chunks,
    then assign whole chunks to splits.

    Why chunks instead of individual files: adjacent video frames are nearly
    identical. If we shuffle frames individually, near-duplicates from the same
    scene end up in both train and val, inflating validation accuracy. Shuffling
    whole chunks keeps temporally close frames in the same split.

    Returns {"train": {class: [paths]}, "val": ..., "test": ...}
    """
    rng    = random.Random(seed)
    splits = {"train": {}, "val": {}, "test": {}}

    for cls, files in files_by_class.items():
        # Group into consecutive chunks (last chunk may be smaller)
        chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
        rng.shuffle(chunks)

        n_chunks  = len(chunks)
        train_end = int(n_chunks * ratios[0])
        val_end   = train_end + int(n_chunks * ratios[1])

        splits["train"][cls] = [f for chunk in chunks[:train_end]        for f in chunk]
        splits["val"][cls]   = [f for chunk in chunks[train_end:val_end] for f in chunk]
        splits["test"][cls]  = [f for chunk in chunks[val_end:]          for f in chunk]

    return splits


def save_split(splits: dict, save_dir: str) -> None:
    """
    Copy files from each split into save_dir/train|val|test/<class>/.
    Uses shutil.copy2 to preserve file metadata; originals are not moved.
    """
    for split_name, class_dict in splits.items():
        for cls, files in class_dict.items():
            dest_dir = os.path.join(save_dir, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for src in files:
                shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))


def build_files_by_class(source_dir: str) -> dict:
    """Public wrapper around _collect_files for use by pipeline modules."""
    return _collect_files(source_dir)


# ---------------------------------------------------------------------------
# Detection-mode helpers (images/ + labels/ flat structure)
# ---------------------------------------------------------------------------

def collect_files_detection(source_dir: str) -> list:
    """
    Return sorted image paths from source_dir/images/.
    Raises FileNotFoundError if the images/ subfolder is missing.
    """
    images_dir = os.path.join(source_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Expected 'images/' subfolder not found in: {source_dir}\n"
            f"Detection datasets must follow the YOLO layout: images/ + labels/"
        )
    return sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def random_split_flat(files: list, ratios: list, seed: int) -> dict:
    """Shuffle a flat file list and split into train/val/test at ratio boundaries."""
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    train, val, test = _cut_at_ratios(shuffled, ratios)
    return {"train": train, "val": val, "test": test}


def chunk_split_flat(files: list, ratios: list, chunk_size: int, seed: int) -> dict:
    """Chunk-shuffle a flat file list and split into train/val/test."""
    rng = random.Random(seed)
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    rng.shuffle(chunks)
    n = len(chunks)
    train_end = int(n * ratios[0])
    val_end   = train_end + int(n * ratios[1])
    return {
        "train": [f for chunk in chunks[:train_end]        for f in chunk],
        "val":   [f for chunk in chunks[train_end:val_end] for f in chunk],
        "test":  [f for chunk in chunks[val_end:]          for f in chunk],
    }


def save_split_detection(splits: dict, save_dir: str, source_dir: str) -> None:
    """
    Copy image + label pairs into save_dir/train|val|test/images and .../labels.
    Labels live at source_dir/labels/<stem>.txt.
    Images without a matching label file are still copied (may be unannotated frames).
    """
    labels_dir = os.path.join(source_dir, "labels")
    for split_name, files in splits.items():
        img_dest = os.path.join(save_dir, split_name, "images")
        lbl_dest = os.path.join(save_dir, split_name, "labels")
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        for src_img in files:
            shutil.copy2(src_img, os.path.join(img_dest, os.path.basename(src_img)))
            stem      = os.path.splitext(os.path.basename(src_img))[0]
            label_src = os.path.join(labels_dir, stem + ".txt")
            if os.path.isfile(label_src):
                shutil.copy2(label_src, os.path.join(lbl_dest, stem + ".txt"))

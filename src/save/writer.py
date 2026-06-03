# Writes augmented images and YOLO label files to disk.
# All output filenames use UUID (uuid1) to guarantee uniqueness across parallel workers.

import os
import uuid
import numpy as np
from PIL import Image


def _save_image(image: np.ndarray, save_dir: str, file_name: str) -> None:
    """Convert a numpy RGB array to JPEG and write it to save_dir/file_name."""
    try:
        Image.fromarray(image).save(os.path.join(save_dir, file_name))
    except Exception:
        pass


def _save_label(labels: list, bboxes: list, save_dir: str, file_name: str) -> None:
    """
    Write YOLO-format annotations to save_dir/<file_name>.txt.
    Each line: <class_id> <cx> <cy> <w> <h>  (6 decimal places, normalized 0–1).
    """
    try:
        label_path = os.path.join(save_dir, file_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for label, bbox in zip(labels, bboxes):
                cx, cy, w, h = bbox
                f.write(f"{int(label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    except Exception:
        pass


def save_file_random(image: np.ndarray, labels: list, bboxes: list, save_dir: str) -> None:
    """
    Save a detection result (image + label file) under save_dir/images/ and save_dir/labels/.
    Filename is a UUID to prevent collisions between parallel workers.
    """
    name      = uuid.uuid1()
    file_name = f"{name}.jpg"
    _save_image(image, os.path.join(save_dir, "images"), file_name)
    _save_label(labels, bboxes, os.path.join(save_dir, "labels"), file_name)


def save_image_random(image: np.ndarray, save_dir: str) -> None:
    """Save a classification image with a UUID filename into save_dir."""
    name = uuid.uuid1()
    _save_image(image, save_dir, f"{name}.jpg")


def save_image_random_part(image: np.ndarray, save_dir: str, part_name: str) -> None:
    """
    Save a frame extracted from video.
    Filename embeds the part name for traceability: <part_name>-<uuid>.jpg
    """
    name = uuid.uuid1()
    _save_image(image, save_dir, f"{part_name}-{name}.jpg")

# Applies a transform to a single dataset entry and returns the result.
# Each function loads the image from disk, runs the transform, and returns
# a tuple ready for the saver — or None if the result is invalid.

import numpy as np
import albumentations as A
from PIL import Image

from augment.image.mixup import mixup
from augment.image.mosaic import mosaic


def detection_transform(target: dict, transform: A.Compose) -> tuple | None:
    """
    Apply an Albumentations detection transform to one image + its bboxes.

    Returns (image, labels, bboxes) on success, or None if:
    - all bboxes are lost after the transform (e.g. rotated out of frame)
    - any exception occurs during loading or transforming
    """
    img = np.array(Image.open(target["image_path"]))
    try:
        result = transform(image=img, bboxes=target["bboxes"], labels=target["labels"])
        if not result["bboxes"]:
            return None
        return result["image"], result["labels"], result["bboxes"]
    except Exception:
        return None


def mixup_transform(target: dict, ex_target: dict) -> tuple | None:
    """
    Blend two detection images using MixUp and merge their bbox lists.
    Returns (image, labels, bboxes), or None if no bboxes remain.
    """
    img    = np.array(Image.open(target["image_path"]))
    ex_img = np.array(Image.open(ex_target["image_path"]))
    try:
        result = mixup(
            image=img,     bboxes=target["bboxes"],    labels=target["labels"],
            ex_image=ex_img, ex_bboxes=ex_target["bboxes"], ex_labels=ex_target["labels"],
        )
        if not result["bboxes"]:
            return None
        return result["image"], result["labels"], result["bboxes"]
    except Exception:
        return None


def mosaic_transform(targets: list, output_size: tuple = (640, 640)) -> tuple | None:
    """
    Combine 4 detection images into a 2×2 mosaic grid.
    targets: list of 4 dataset dicts, each with image_path/bboxes/labels.
    Returns (image, labels, bboxes), or None if no bboxes survive clipping.
    """
    try:
        images      = [np.array(Image.open(t["image_path"])) for t in targets]
        bboxes_list = [t["bboxes"] for t in targets]
        labels_list = [t["labels"] for t in targets]

        result = mosaic(images, bboxes_list, labels_list, output_size)
        if not result["bboxes"]:
            return None
        return result["image"], result["labels"], result["bboxes"]
    except Exception:
        return None


def classification_transform(target: dict, transform: A.Compose) -> tuple | None:
    """
    Apply a classification transform to one image (no bboxes).
    Returns (image, save_dir) so the caller knows where to write the output.
    """
    img = np.array(Image.open(target["image_path"]))
    try:
        result = transform(image=img)
        return result["image"], target["save_dir"]
    except Exception:
        return None

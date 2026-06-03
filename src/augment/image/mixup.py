# MixUp augmentation for object detection.
# Blends two images by alpha-compositing them and merges their bbox lists.
# Both images are zero-padded to the same canvas size before blending.

import numpy as np


def mixup(
    image: np.ndarray, bboxes: list, labels: list,
    ex_image: np.ndarray, ex_bboxes: list, ex_labels: list,
) -> dict:
    """
    Blend two images with a random alpha in [0.6, 0.7] and merge annotations.

    When the two images have different sizes, both are padded with zeros to
    the maximum of each dimension — this preserves all existing bboxes without
    any rescaling artefact.

    Bbox coordinates are then scaled to the padded canvas so they remain
    correct relative to the blended output.

    Returns:
        {"image": blended ndarray, "bboxes": merged list, "labels": merged list}
    """
    h1, w1 = image.shape[:2]
    h2, w2 = ex_image.shape[:2]
    H, W = max(h1, h2), max(w1, w2)

    # Pad both images to the same (H, W) canvas
    c1 = np.pad(image,    ((0, H - h1), (0, W - w1), (0, 0)), "constant")
    c2 = np.pad(ex_image, ((0, H - h2), (0, W - w2), (0, 0)), "constant")

    # Scale YOLO-normalized bboxes to account for the padding
    bboxes    = [list(b) for b in bboxes]
    ex_bboxes = [list(b) for b in ex_bboxes]

    for b in bboxes:
        b[0] *= w1 / W   # cx
        b[1] *= h1 / H   # cy
        b[2] *= w1 / W   # w
        b[3] *= h1 / H   # h

    for b in ex_bboxes:
        b[0] *= w2 / W
        b[1] *= h2 / H
        b[2] *= w2 / W
        b[3] *= h2 / H

    lam   = np.random.uniform(0.6, 0.7)
    mixed = np.clip(lam * c1 + (1 - lam) * c2, 0, 255).astype(np.uint8)

    return {
        "image":  mixed,
        "bboxes": bboxes + ex_bboxes,
        "labels": labels + ex_labels,
    }

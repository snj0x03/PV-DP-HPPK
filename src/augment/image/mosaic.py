# Mosaic augmentation for object detection (popularized by YOLOv4).
# Combines 4 images into a single 2×2 grid with a random center point.
# Each image is resized to fill its quadrant; bboxes are remapped accordingly.

import random
import cv2
import numpy as np


def mosaic(
    images: list,
    bboxes_list: list,
    labels_list: list,
    output_size: tuple = (640, 640),
    min_bbox_area: float = 0.01,
) -> dict:
    """
    Combine 4 images into a 2×2 mosaic on a canvas of output_size (H, W).

    A random center point (cx, cy) divides the canvas into four quadrants.
    Each image is resized to fill its quadrant, and its YOLO-normalized bboxes
    are remapped to the new absolute position, then re-normalized to output_size.
    Boxes that become too small (< min_bbox_area in both w and h) are dropped.

    Args:
        images:        list of 4 RGB numpy arrays (any size)
        bboxes_list:   list of 4 bbox lists  [[cx,cy,w,h], ...]  YOLO normalized
        labels_list:   list of 4 label lists  [int, ...]
        output_size:   (H, W) of the output canvas
        min_bbox_area: minimum normalized side length to keep a bbox

    Returns:
        {"image": canvas ndarray, "bboxes": list, "labels": list}
    """
    H, W = output_size

    # Random center kept away from edges so no quadrant is degenerate
    cx = int(random.uniform(0.3 * W, 0.7 * W))
    cy = int(random.uniform(0.3 * H, 0.7 * H))

    canvas     = np.zeros((H, W, 3), dtype=np.uint8)
    all_bboxes = []
    all_labels = []

    # Canvas regions (x1,y1,x2,y2) for each quadrant
    regions = [
        (0,  0,  cx, cy),   # top-left
        (cx, 0,  W,  cy),   # top-right
        (0,  cy, cx, H),    # bottom-left
        (cx, cy, W,  H),    # bottom-right
    ]

    for img, bboxes, labels, (x1, y1, x2, y2) in zip(images, bboxes_list, labels_list, regions):
        rw = x2 - x1   # quadrant width
        rh = y2 - y1   # quadrant height

        canvas[y1:y2, x1:x2] = cv2.resize(img, (rw, rh))

        for bbox, label in zip(bboxes, labels):
            bcx, bcy, bw, bh = bbox  # normalized to original image

            # Step 1: scale to absolute coords within this quadrant
            abs_cx = bcx * rw + x1
            abs_cy = bcy * rh + y1
            abs_w  = bw  * rw
            abs_h  = bh  * rh

            # Step 2: normalize to full output canvas
            norm_cx = abs_cx / W
            norm_cy = abs_cy / H
            norm_w  = abs_w  / W
            norm_h  = abs_h  / H

            # Clip center; shrink width/height so box stays within canvas
            norm_cx = float(np.clip(norm_cx, 0.0, 1.0))
            norm_cy = float(np.clip(norm_cy, 0.0, 1.0))
            norm_w  = min(float(norm_w), 2 * min(norm_cx, 1.0 - norm_cx))
            norm_h  = min(float(norm_h), 2 * min(norm_cy, 1.0 - norm_cy))

            if norm_w >= min_bbox_area and norm_h >= min_bbox_area:
                all_bboxes.append([norm_cx, norm_cy, norm_w, norm_h])
                all_labels.append(label)

    return {"image": canvas, "bboxes": all_bboxes, "labels": all_labels}

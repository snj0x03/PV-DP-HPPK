# Reads a YOLO-format detection dataset (images/ + labels/) into memory.
# Each entry holds the image path plus its parsed bounding boxes and class IDs.

import os


def yolo_dataset(file_dir: str) -> list[dict]:
    """
    Load every image from file_dir/images/ and its paired YOLO label from
    file_dir/labels/.

    YOLO label format (one object per line):
        <class_id> <cx> <cy> <w> <h>   (all values normalized 0–1)

    Returns a list of dicts:
        {"image_path": ..., "labels": [int, ...], "bboxes": [[cx,cy,w,h], ...]}

    Images whose label file is missing or unreadable are silently skipped.
    """
    image_dir = os.path.join(file_dir, "images")
    label_dir = os.path.join(file_dir, "labels")
    dataset = []

    for file_name in os.listdir(image_dir):
        if not file_name.endswith(".jpg"):
            continue
        try:
            image_path = os.path.join(image_dir, file_name)
            label_path = os.path.join(label_dir, file_name.replace(".jpg", ".txt"))

            labels, bboxes = [], []
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.split()
                    labels.append(int(parts[0]))
                    bboxes.append([float(v) for v in parts[1:]])

            dataset.append({
                "image_path": image_path,
                "labels":     labels,
                "bboxes":     bboxes,
            })
        except Exception:
            pass

    return dataset

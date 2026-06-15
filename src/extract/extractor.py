import os
import cv2
import numpy as np


def extract_image(file_dir: str, sub_dir: str, file_name: str) -> np.ndarray | None:
    image_path = os.path.join(file_dir, sub_dir, file_name)
    image = cv2.imread(image_path)
    return image


def extract_label(file_dir: str, sub_dir: str, file_name: str) -> tuple:
    label_path = os.path.join(file_dir, sub_dir, file_name)
    label_path = label_path.replace(".jpg", ".txt")

    labels, bboxes = [], []
    with open(label_path, "r") as f:
        for lines in f.readlines():
            label, boxes = lines.split()[0], lines.split()[1:]
            label = int(label)
            boxes = [float(box) for box in boxes]
            labels.append(label)
            bboxes.append(boxes)
    return labels, bboxes

def extract_target(file_dir: str, file_name: str) -> dict | None:
    try:
        image = extract_image(file_dir, "images", file_name)
        labels, bboxes = extract_label(file_dir, "labels", file_name)

        return {
            "image": image,
            "labels": labels,
            "bboxes": bboxes
        }
    except:
        pass

def extract_mixup_metadata(file_dir: str, mixup_file_name: list[str]):
    metadata = []
    for file_name in mixup_file_name:
        target = extract_target(file_dir, file_name)
        if target:
            temp = {
                "image": target["image"],
                "bboxes": target["bboxes"],
                "bbox_labels": {"labels": target["labels"]}
            }
            
            metadata.append(temp)
    return metadata
        
def extract_mosaic_metadata(file_dir: str, mosaic_file_name: list[str]):
    metadata = []
    for file_name in mosaic_file_name:
        target = extract_target(file_dir, file_name)
        if target:
            temp = {
                "image": target["image"],
                "bboxes": target["bboxes"],
                "bbox_labels": {"labels": target["labels"]}
            }
            
            metadata.append(temp)
    return metadata

import os
import cv2
import uuid
import numpy as np


def save_image(image: np.ndarray, save_dir: str, sub_dir: str, file_name: str) -> None:
    try:
        if sub_dir:
            save_dir = os.path.join(save_dir, sub_dir)

        cv2.imwrite(os.path.join(save_dir, file_name), image)
        
        # Image.fromarray(image).save(os.path.join(save_dir, file_name))
    except:
        pass


def save_label(labels: list, bboxes: list, save_dir: str, sub_dir: str, file_name: str) -> None:
    try:
        if sub_dir:
            save_dir = os.path.join(save_dir, sub_dir)
        label_name = file_name.replace(".jpg", ".txt")
        with open(os.path.join(save_dir, label_name), "w") as f:
            for label, bbox in zip(labels, bboxes):
                cx, cy, w, h = bbox
                f.write(f"{int(label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    except:
        pass


def save_file_random(image: np.ndarray, labels: list, bboxes: list, save_dir: str) -> None:
        name = uuid.uuid1()
        file_name = f"{name}.jpg"
        text_name = f"{name}.txt"
        save_image(image, save_dir, "images", file_name)
        save_label(labels, bboxes, save_dir, "labels", text_name)


def save_image_random(image: np.ndarray, save_dir: str, sub_dir: str) -> None:
    name = uuid.uuid1()
    file_name = f"{name}.jpg"
    save_image(image, save_dir, sub_dir, file_name)


def save_image_random_part(image: np.ndarray, save_dir: str, sub_dir: str, part_name: str) -> None:
    name = uuid.uuid1()
    file_name = f"{part_name}-{name}.jpg"
    save_image(image, save_dir, sub_dir, file_name)


def save_result(result: tuple | None, save_dir: str):
    if result:
        image, labels, bboxes = result
        save_file_random(image, labels, bboxes, save_dir) 



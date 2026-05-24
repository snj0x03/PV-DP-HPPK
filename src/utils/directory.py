import os
from PIL import Image
import numpy as np
import pandas as pd


def save_image(aug_img: np.ndarray, img_save_dir: str, aug_name: str) -> None:
    Image.fromarray(aug_img).save(os.path.join(img_save_dir, aug_name))


def save_label(aug_labels: list, aug_bboxes: list, lbl_save_dir: str, aug_name: str) -> None:
    lbl_name = aug_name.replace(".jpg", ".txt")
    with open(os.path.join(lbl_save_dir, lbl_name), "w") as f:
        for label, bbox in zip(aug_labels, aug_bboxes):
            cx, cy, w, h = bbox
            f.write(f"{int(label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def create_save_dirs(yolo_save_dir: str) -> tuple[str, str]:
    img_save_dir = os.path.join(yolo_save_dir, "images")
    lbl_save_dir = os.path.join(yolo_save_dir, "labels")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(lbl_save_dir, exist_ok=True)
    return img_save_dir, lbl_save_dir

def setup_save_dir(video_dir: str, frame_save_dir: str) -> None:
    for directory in os.listdir(video_dir):
        if os.path.isdir(directory):
            os.makedirs(os.path.join(frame_save_dir, directory))



def load_part_names(csv_path: str) -> dict[str, str]:
    df = pd.read_csv(csv_path, header=0)
    part_names = {} 
    for _, row in df.iterrows():
        part_names[str(row[0])] = str(row[1])

    return part_names 
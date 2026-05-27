import os
from PIL import Image
import numpy as np
import pandas as pd


def save_image(img: np.ndarray, save_dir: str, filename: str) -> None:
    Image.fromarray(img).save(os.path.join(save_dir, filename))


def save_label(labels: list, bboxes: list, save_dir: str, filename: str) -> None:
    lbl_name = filename.replace(".jpg", ".txt")
    with open(os.path.join(save_dir, lbl_name), "w") as f:
        for label, bbox in zip(labels, bboxes):
            cx, cy, w, h = bbox
            f.write(f"{int(label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def create_save_dir(main_dir: str, save_dir: str) -> None:
    for dirname in os.listdir(main_dir):
        if os.path.isdir(os.path.join(main_dir, dirname)):
            os.makedirs(os.path.join(save_dir, dirname))


def load_part_names(csv_path: str) -> dict[str, str]:
    df = pd.read_csv(csv_path, header=0)
    part_names = {} 
    for _, row in df.iterrows():
        part_names[str(row[0])] = str(row[1])

    return part_names 


import numpy as np 
from PIL import Image
import os
from utils.directory import create_save_dirs, save_image, save_label
from Augmentation.Custom import MixUp


def apply_transform(target, pipeline_transform, yolo_save_dir, image_pool, augmentation_list):
    img = np.array(Image.open(target["image_path"]))

    result     = pipeline_transform(image=img, bboxes=target["bboxes"], labels=target["labels"])
    aug_img    = result["image"]
    aug_bboxes = result["bboxes"]
    aug_labels = result["labels"]

    if not aug_bboxes:
        return

    stem     = os.path.splitext(os.path.basename(target["image_path"]))[0]
    aug_name = f"{stem}_aug.jpg"

    img_save_dir, lbl_save_dir = create_save_dirs(yolo_save_dir)

    save_image(aug_img, img_save_dir, aug_name)
    save_label(aug_labels, aug_bboxes, lbl_save_dir, aug_name)

    
def apply_mixup(target1, target2, yolo_save_dir):
    result = MixUp(target1, target2)
    img = result["image"]
    bboxes = result["bboxes"]
    labels = result["labels"]
    save_image(img, yolo_save_dir)
    save_label(labels, bboxes, yolo_save_dir)




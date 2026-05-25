import numpy as np 
from PIL import Image
import os
import uuid
from utils.directory import save_image, save_label
from Augmentation.Custom import MixUp
import logging

logging.basicConfig(filename="log.log", level=logging.INFO)


def apply_transform(target, pipeline_transform, save_dir):
    img = np.array(Image.open(target["image_path"]))

    result     = pipeline_transform(image=img, bboxes=target["bboxes"], labels=target["labels"])
    image    = result["image"]
    bboxes = result["bboxes"]
    labels = result["labels"]

    logging.info(f"\n{target["image_path"]}\n{bboxes}\n{labels}")
    if not bboxes:
        return

    # stem = os.path.splitext(os.path.basename(target["image_path"]))[0]
    name = uuid.uuid1()
    filename = f"{name}.jpg"
    textname = f"{name}.txt"

    save_image(image, os.path.join(save_dir, "images"), filename)
    save_label(labels, bboxes, os.path.join(save_dir, "labels"), textname)

    
def apply_mixup(target1, target2, save_dir):

    img = np.array(Image.open(target1["image_path"]))
    ex_img = np.array(Image.open(target2["image_path"]))

    result = MixUp(image = img,
                   bboxes = target1["bboxes"],
                   labels = target1["labels"],
                   ex_image = ex_img,
                   ex_bboxes = target2["bboxes"],
                   ex_labels = target2["labels"])

    
    image = result["image"]
    bboxes = result["bboxes"]
    labels = result["labels"]
    logging.info(f"\n{target1["image_path"]}\n{bboxes}\n{labels}")

    if not bboxes:
        return

    name = uuid.uuid1()
    filename = f"{name}m1x.jpg"
    textname = f"{name}m1x.txt"
    save_image(image, os.path.join(save_dir, "images"), filename)
    save_label(labels, bboxes, os.path.join(save_dir, "labels"), textname)




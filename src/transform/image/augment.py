import numpy as np 
import albumentations as A
from PIL import Image
from transform.image.custom import mixup 


def detection_transform(target: dict, transform: A.Compose) -> tuple | None:
    img = np.array(Image.open(target["image_path"]))
    try:
        result = transform(image=img, bboxes=target["bboxes"], labels=target["labels"])
        image = result["image"]
        bboxes = result["bboxes"]
        labels = result["labels"]

   
        if not bboxes:
            return

        return image, labels, bboxes 

    except:
        pass


def mixup_transform(target: dict, ex_target: dict) -> tuple | None:
    img = np.array(Image.open(target["image_path"]))
    ex_img = np.array(Image.open(ex_target["image_path"]))
    try:
        result = mixup(image = img,
                    bboxes = target["bboxes"],
                    labels = target["labels"],
                    ex_image = ex_img,
                    ex_bboxes = ex_target["bboxes"],
                    ex_labels = ex_target["labels"])
        image = result["image"]
        bboxes = result["bboxes"]
        labels = result["labels"]

        if not bboxes:
            return

        return image, labels, bboxes

    except:
        pass


def classification_transform(target: dict, transform: A.Compose) -> tuple | None:
    img = np.array(Image.open(target["image_path"]))
    try:
        result = transform(image=img)
        image = result["image"]
        image_save_dir = target["save_dir"]

        return image, image_save_dir 

    except:
        pass





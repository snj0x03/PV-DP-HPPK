import albumentations as A
import numpy as np
from transform.image.custom import mixup 
from transform.image.default import geometric 


# Object Detection Transoforms

def detection_transform(target: dict, transform: A.Compose) -> tuple | None:
    if not target:
        return 
    try:
        result = transform(image=target["image"], bboxes=target["bboxes"], labels=target["labels"])
        image = result["image"]
        bboxes = result["bboxes"]
        labels = result["labels"]

        return image, labels, bboxes

    except:
        pass


def mixup_transform(target: dict, ex_target: dict) -> tuple | None:
    try:
        # Rotate
        target = geometric(image=target["image"], bboxes=target["bboxes"], labels=target["labels"])

        result = mixup(image = target["image"],
                    bboxes = target["bboxes"],
                    labels = target["labels"],
                    ex_image = ex_target["image"],
                    ex_bboxes = ex_target["bboxes"],
                    ex_labels = ex_target["labels"])
        image = result["image"]
        bboxes = result["bboxes"]
        labels = result["labels"]

        return image, labels, bboxes

    except:
        pass


def mosaic_transform(target: dict, transform: A.Compose, metadata: list) -> tuple | None:
    try:
        result = transform(image = target["image"],
                           bboxes = target["bboxes"],
                           labels = target["labels"],
                           mosaic_metadata = metadata)
        image = result["image"]
        bboxes = result["bboxes"]
        labels = result["labels"]

        return image, labels, bboxes
    except:
        pass


# Classification Transforms

def classification_transform(target: np.ndarray | None, transform: A.Compose) -> np.ndarray | None:
    try:
        result = transform(image=target)
        image = result["image"]
        return image
    except:
        pass

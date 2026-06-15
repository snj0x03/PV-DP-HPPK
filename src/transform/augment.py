import albumentations as A
import numpy as np


# Object Detection Transforms
def detection_transform(target: dict, transform: A.Compose, mixup_metadata: list | None, mosaic_metadata: list | None) -> tuple | None:
    try:
        result = transform(image=target["image"], 
                           bboxes=target["bboxes"], 
                           labels=target["labels"],
                           mixup_metadata=mixup_metadata,
                           mosaic_metadata=mosaic_metadata)
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

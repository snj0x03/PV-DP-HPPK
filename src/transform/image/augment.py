import albumentations as A
from transform.image.custom import mixup 
from transform.image.default import geometric 

def detection_transform(target: dict, transform: A.Compose) -> tuple | None:
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


def classification_transform(target: dict, transform: A.Compose) -> tuple | None:
    try:
        result = transform(image=target["image"])
        image = result["image"]
        label = target["label"]

        return image, label 

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

def copypaste_transform(target: dict, transform: A.Compose, metadata: list) -> tuple | None:
    try:
        result = transform(image = target["image"],
                           bboxes = target["bboxes"],
                           labels = target["labels"],
                           copy_paste_metadata = metadata)
        image = result["image"]
        bboxes = result["bboxes"]
        labels = result["labels"]
        
        return image, labels, bboxes

    except:
        pass


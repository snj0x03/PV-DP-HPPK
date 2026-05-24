# Utils

import os
import random
import numpy as np
from PIL import Image
import albumentations as A


def augmentation_pipeline(augmentation_list):
    pipeline_transform = A.Compose(
        [
            A.HorizontalFlip(p=augmentation_list["horizontal_flip"]["prod"]),
            A.VerticalFlip(p=augmentation_list["vertical_flip"]["prod"]),
            A.Rotate(limit=15, p=augmentation_list["rotate"]["prod"]),
            A.GaussianBlur(blur_limit=3, p=augmentation_list["blur"]["prod"]),
            A.GaussNoise(p=augmentation_list["noise"]["prod"]),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )
    return pipeline_transform


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


def apply_mixup(target, img, image_pool, stem, img_save_dir, lbl_save_dir, prod):
    if random.random() >= prod or len(image_pool) < 2:
        return

    other  = random.choice([t for t in image_pool if t["image_path"] != target["image_path"]])
    img2   = np.array(Image.open(other["image_path"]))
    lam    = np.random.beta(0.35, 0.35)
    h, w   = img.shape[:2]
    img2   = np.array(Image.fromarray(img2).resize((w, h)))
    mixed  = np.clip(lam * img + (1 - lam) * img2, 0, 255).astype(np.uint8)

    merged_labels = target["labels"] + other["labels"]
    merged_bboxes = target["bboxes"] + other["bboxes"]

    if not merged_bboxes:
        return

    mix_name = f"{stem}_mixup.jpg"
    save_image(mixed, img_save_dir, mix_name)
    save_label(merged_labels, merged_bboxes, lbl_save_dir, mix_name)


def apply_mosaic(target, img, image_pool, stem, img_save_dir, lbl_save_dir, prod):
    if random.random() >= prod or len(image_pool) < 4:
        return

    others = random.sample([t for t in image_pool if t["image_path"] != target["image_path"]], 3)
    mosaic_meta = [
        {
            "image" : np.array(Image.open(t["image_path"])),
            "bboxes": t["bboxes"],
            "labels": t["labels"],
        }
        for t in others
    ]

    mosaic_compose = A.Compose(
        [A.Mosaic(grid_yx=(2, 2), target_size=(512, 512), cell_shape=(512, 512), p=1.0)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], min_visibility=0.3),
    )

    result = mosaic_compose(
        image=img,
        bboxes=target["bboxes"],
        labels=target["labels"],
        mosaic_metadata=mosaic_meta,
    )

    if not result["bboxes"]:
        return

    mos_name = f"{stem}_mosaic.jpg"
    save_image(result["image"], img_save_dir, mos_name)
    save_label(result["labels"], result["bboxes"], lbl_save_dir, mos_name)


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

    apply_mixup(target, img, image_pool, stem, img_save_dir, lbl_save_dir, augmentation_list["mixup"]["prod"])
    apply_mosaic(target, img, image_pool, stem, img_save_dir, lbl_save_dir, augmentation_list["mosaic"]["prod"])
    

def yolo_dataset(yolo_dir: str) -> list[dict[str, str | list[int] | list[list[int]]]]:
    image_dir = os.path.join(yolo_dir, "images")
    label_dir = os.path.join(yolo_dir, "labels")
    dataset = []
    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg")):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

            labels, bboxes = [], []
            with open(label_path, "r") as f:
                for lines in f.readlines():
                    label, boxes = lines.split()[0], lines.split()[1:]
                    label = int(label)
                    boxes = [float(box) for box in boxes]
                    labels.append(label)
                    bboxes.append(boxes)

            target = {
                "image_path": image_path, 
                "labels": labels,
                "bboxes": bboxes 
            }
            dataset.append(target)

    return dataset


def run_augment(yolo_dir: str, yolo_save_dir: str, augmentation_list: dict[str, dict[str, str | int]]) -> None:

    dataset            = yolo_dataset(yolo_dir)
    pipeline_transform = augmentation_pipeline(augmentation_list)
    for target in dataset:
        apply_transform(target, pipeline_transform, yolo_save_dir, dataset, augmentation_list)

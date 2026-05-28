from Dataset.yolo import yolo_dataset
from Dataset.video import image_dataset
from Augmentation.Pipeline import augment_transform, copy_transform
from Augmentation.transform import apply_transform, apply_mixup, apply_transform2
from utils.directory import create_save_dir
import random


def run_augment(file_dir: str, save_dir: str, copy: bool, multiplier: int, mixup: bool) -> None:

    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")

    dataset = yolo_dataset(file_dir)
    print("Yolo Dataset Created")

    cnt = 0
    for target in dataset:
        if copy == True:
            apply_transform(target, copy_transform, save_dir)
        for _ in range(multiplier): 
            apply_transform(target, augment_transform, save_dir)
        if mixup == True: 
            apply_mixup(target, random.choice(dataset), save_dir)

        cnt += 1
        print(f"Progress - {int(cnt*100 / len(dataset))}% - {cnt}/{len(dataset)}", end="\r", flush=True)
    print("Augmentation Completed")

def run_augment2(file_dir, save_dir, copy, multiplier):
    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")

    dataset = image_dataset(file_dir, save_dir)
    print("Image Dataset Created")

    cnt = 0
    for target in dataset:
        if copy == True:
            apply_transform2(target, copy_transform)
        for _ in range(multiplier):
            apply_transform2(target, augment_transform)

        cnt += 1
        print(f"Progress - {int(cnt*100 / len(dataset))}% - {cnt}/{len(dataset)}", end="\r", flush=True)
    print("Augmentation Completed")

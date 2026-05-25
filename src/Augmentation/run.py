from Dataset.yolo import yolo_dataset
from Augmentation.Pipeline import pipeline_transform
from Augmentation.transform import apply_transform, apply_mixup
from utils.directory import create_save_dir
import random


def run_augment(file_dir: str, save_dir: str, multiplier, mixup) -> None:

    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")
    dataset = yolo_dataset(file_dir)
    print("Yolo Dataset Created")

    cnt = 0
    for target in dataset:
        for _ in range(multiplier): 
            apply_transform(target, pipeline_transform, save_dir)
        if mixup == True: 
            apply_mixup(target, random.choice(dataset), save_dir)

        cnt += 1
        print(f"Progress - {int(cnt*100 / len(dataset))}% - {cnt}/{len(dataset)}", end="\r", flush=True)
    print("Augmentation Completed")


from Dataset.yolo import yolo_dataset
from Augmentation.Pipeline import pipeline_transform
from Augmentation.transform import apply_transform
from run_augmentation import apply_mixup




def run_augment(yolo_dir: str, yolo_save_dir: str, augmentation_list: dict[str, dict[str, str | int]]) -> None:

    dataset            = yolo_dataset(yolo_dir)
    pipeline_transform = pipeline_transform
    for target in dataset:
        apply_transform(target, pipeline_transform, yolo_save_dir, dataset, augmentation_list)
        apply_mixup(target, random.choice(dataset)) # implement random


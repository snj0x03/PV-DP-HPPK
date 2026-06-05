# Classification augmentation pipeline.
# Reads a class-per-folder image dataset and generates augmented variants in parallel.
# No bounding boxes are involved — images are augmented and saved directly.

import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

from reader.image import image_dataset
from augment.image.presets import classify_transform, empty_transform
from augment.image.apply import classification_transform
from utils.helpers import create_save_dir
from save.writer import save_image_random


def process_target(args: dict, copy: bool, multiplier: int) -> None:
    """
    Worker function for classification augmentation.

    Steps:
    - If copy=True: save the original image unchanged (empty_transform is a no-op).
    - For each of multiplier iterations: apply random augmentation and save.
    """
    target = args

    if copy:
        result = classification_transform(target, empty_transform)
        if result:
            image, image_save_dir = result
            save_image_random(image, image_save_dir)

    for _ in range(multiplier):
        result = classification_transform(target, classify_transform)
        if result:
            image, image_save_dir = result
            save_image_random(image, image_save_dir)


def classification_pipeline(file_dir: str, save_dir: str, copy: bool, multiplier: int) -> None:
    """
    Full classification augmentation pipeline:
    1. Mirror source class-folder structure under save_dir.
    2. Collect all images with their target output directories.
    3. Augment and save in parallel across all CPU cores.
    """
    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")

    dataset = image_dataset(file_dir, save_dir)
    print("Image Dataset Created")

    f = partial(process_target, copy=copy, multiplier=multiplier)

    print("Augmentation Process Initiated")
    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(f, dataset), total=len(dataset), desc="Progress"):
            pass
    print("Augmentation Completed")

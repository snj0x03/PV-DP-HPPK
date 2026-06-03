# Detection augmentation pipeline.
# Reads a YOLO-format dataset and generates augmented variants in parallel.
# Supports three augmentation strategies: standard transforms, MixUp, and Mosaic.

import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

from reader.detection import yolo_dataset
from augment.image.presets import augment_transform, copy_transform
from augment.image.apply import detection_transform, mixup_transform, mosaic_transform
from utils.helpers import create_save_dir, make_pair_list, make_mosaic_list
from save.writer import save_file_random


def process_target(args: list, copy: bool, multiplier: int, mixup: bool, save_dir: str) -> None:
    """
    Worker function for standard augmentation + optional MixUp.
    args: [target, random_partner]  — partner is used only when mixup=True.

    Steps:
    - If copy=True: save the original image unchanged (copy_transform is a no-op).
    - For each of multiplier iterations: apply random augment and save.
    - If mixup=True: blend target with its random partner and save.
    """
    target, random_target = args

    if copy:
        result = detection_transform(target, copy_transform)
        if result:
            save_file_random(*result, save_dir)

    for _ in range(multiplier):
        result = detection_transform(target, augment_transform)
        if result:
            save_file_random(*result, save_dir)

    if mixup:
        result = mixup_transform(target, random_target)
        if result:
            save_file_random(*result, save_dir)


def process_mosaic(group: list, save_dir: str) -> None:
    """
    Worker function for Mosaic augmentation.
    group: list of 4 dataset dicts — combined into one 2×2 image.
    """
    result = mosaic_transform(group)
    if result:
        save_file_random(*result, save_dir)


def detection_pipeline(
    file_dir: str,
    save_dir: str,
    copy: bool,
    multiplier: int,
    mixup: bool,
    mosaic: bool = False,
) -> None:
    """
    Full detection augmentation pipeline:
    1. Mirror source folder structure (images/ + labels/) under save_dir.
    2. Load all annotated images from file_dir.
    3. Run standard augmentation + MixUp in parallel across all CPU cores.
    4. If mosaic=True, run a second parallel pass for Mosaic augmentation.
    """
    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")

    dataset = yolo_dataset(file_dir)
    print("Detection Dataset Created")

    # Pre-generate random pairs for MixUp before spawning workers
    tasks = make_pair_list(dataset)

    f = partial(process_target, copy=copy, multiplier=multiplier, mixup=mixup, save_dir=save_dir)
    print("Augmentation Process Initiated")
    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(f, tasks), total=len(tasks), desc="Augment"):
            pass

    if mosaic:
        # Pre-generate groups of 4 for Mosaic before spawning workers
        mosaic_tasks = make_mosaic_list(dataset)
        f_mosaic = partial(process_mosaic, save_dir=save_dir)
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f_mosaic, mosaic_tasks), total=len(mosaic_tasks), desc="Mosaic"):
                pass

    print("Augmentation Completed")

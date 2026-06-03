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


def process_target(args: list, copy: bool, multiplier: int, mixup: bool, save_dir: str) -> tuple:
    """
    Worker function for standard augmentation + optional MixUp.
    args: [target, random_partner]  — partner is used only when mixup=True.

    Returns (saved, attempted) so the caller can report drop statistics.
    A result is dropped when all bboxes are lost after the transform or an
    exception occurs (e.g. corrupt image, degenerate bbox coordinates).
    """
    target, random_target = args
    saved = attempted = 0

    if copy:
        attempted += 1
        result = detection_transform(target, copy_transform)
        if result:
            save_file_random(*result, save_dir)
            saved += 1

    for _ in range(multiplier):
        attempted += 1
        result = detection_transform(target, augment_transform)
        if result:
            save_file_random(*result, save_dir)
            saved += 1

    if mixup:
        attempted += 1
        result = mixup_transform(target, random_target)
        if result:
            save_file_random(*result, save_dir)
            saved += 1

    return saved, attempted


def process_mosaic(group: list, save_dir: str) -> tuple:
    """
    Worker function for Mosaic augmentation.
    group: list of 4 dataset dicts — combined into one 2×2 image.
    Returns (saved, attempted).
    """
    result = mosaic_transform(group)
    if result:
        save_file_random(*result, save_dir)
        return 1, 1
    return 0, 1


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
    total_saved = total_attempted = 0
    with Pool(processes=cpu_count()) as pool:
        with tqdm.tqdm(total=len(tasks), desc="Augment") as pbar:
            for saved, attempted in pool.imap_unordered(f, tasks):
                total_saved    += saved
                total_attempted += attempted
                pbar.update(1)

    if mosaic:
        mosaic_tasks = make_mosaic_list(dataset)
        f_mosaic = partial(process_mosaic, save_dir=save_dir)
        with Pool(processes=cpu_count()) as pool:
            with tqdm.tqdm(total=len(mosaic_tasks), desc="Mosaic") as pbar:
                for saved, attempted in pool.imap_unordered(f_mosaic, mosaic_tasks):
                    total_saved    += saved
                    total_attempted += attempted
                    pbar.update(1)

    dropped = total_attempted - total_saved
    if dropped:
        pct = dropped / total_attempted * 100 if total_attempted else 0
        print(f"Augmentation Completed — {total_saved}/{total_attempted} saved "
              f"({dropped} dropped, {pct:.1f}%: bboxes lost or transform error)")
    else:
        print(f"Augmentation Completed — {total_saved}/{total_attempted} saved")

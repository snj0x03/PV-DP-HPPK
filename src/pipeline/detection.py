import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from dataset.yolo import yolo_dataset
from transform.image.default import augment_transform, copy_transform 
from transform.image.augment import detection_transform, mixup_transform
from utils.helper import create_save_dir, create_tasks 
from extract.extractor import extract_target, extract_mosaic_metadata
from load.loader import save_result
from transform.image.augment import mosaic_transform 
from transform.image.default import mosaic 


def process_augment(args: dict, file_dir: str, save_dir: str) -> None:

    file_name = args[0]

    target = extract_target(file_dir, file_name) # Extract
    if target: 
        result = detection_transform(target, augment_transform) # Transform
        save_result(result, save_dir) # Load
    return

def process_copy(args: dict, file_dir: str, save_dir: str) -> None:

    file_name = args[0]

    target = extract_target(file_dir, file_name) # Extract
    if target:
        result = detection_transform(target, copy_transform) # Transform
        save_result(result, save_dir) # Load

    return

def process_mixup(args: list, file_dir: str, save_dir: str) -> None:

    file_name = args[0]
    ex_file_name = args[1]

    target = extract_target(file_dir, file_name) # Extract
    ex_target = extract_target(file_dir, ex_file_name) # Extract
    if target and ex_target:
        result = mixup_transform(target, ex_target) # Transform
        save_result(result, save_dir) # Load
    return 

def process_mosaic(args: list, file_dir:str, save_dir: str) -> None:

    file_name = args[0]
    mosaic_file_name = args[1:]

    mosaic_metadata = extract_mosaic_metadata(file_dir, mosaic_file_name)
    target = extract_target(file_dir, file_name) # Extract
    if target:
        result = mosaic_transform(target, mosaic, mosaic_metadata) # Transform
        save_result(result, save_dir) # Load
    return 

# Pipeline
def detection_pipeline(file_dir: str, 
                       save_dir: str, 
                       mode: str,
                       aug_mult: int, 
                       copy_mult: bool, 
                       mixup_mult: bool,
                       mosaic_mult: bool) -> None:

    create_save_dir(file_dir, save_dir, mode)
    print("Save Directory Created")

    # Dataset 
    dataset = yolo_dataset(file_dir)

    # Augmentation
    if aug_mult: print("Augmentation Initiated")
    for i in range(aug_mult):
        tasks = create_tasks(dataset, 1)
        f_aug = partial(process_augment, file_dir=file_dir, save_dir = save_dir)
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f_aug, tasks), total=len(tasks), desc=f"Iter {i+1}"):
                pass

    # Copy 
    if copy_mult: print("Copy Initiated")
    for i in range(copy_mult):
        tasks = create_tasks(dataset, 1)
        f_copy = partial(process_copy, file_dir=file_dir, save_dir=save_dir)
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f_copy, tasks), total=len(tasks), desc=f"Iter {i+1}"):
                pass

    # Mixup
    if mixup_mult: print("Mixup Augmentation Initiated")
    for i in range(mixup_mult):
        tasks = create_tasks(dataset, 2)
        f_mixup = partial(process_mixup, file_dir=file_dir, save_dir=save_dir)
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f_mixup, tasks), total=len(tasks), desc=f"Iter {i+1}"):
                pass

    # Mosaic
    if mosaic_mult: print("Mosaic Augmentation Initiated")
    for i in range(mosaic_mult):
        tasks = create_tasks(dataset, 4)
        f_mosaic = partial(process_mosaic, file_dir=file_dir, save_dir=save_dir)
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f_mosaic, tasks), total=len(tasks), desc=f"Iter {i+1}"):
                pass

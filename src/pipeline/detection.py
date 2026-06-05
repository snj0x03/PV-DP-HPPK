import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from dataset.yolo import yolo_dataset
from transform.image.default import augment_transform, copy_transform 
from transform.image.augment import detection_transform, mixup_transform
from utils.helper import create_save_dir, create_mixup_tasks, create_mosaic_tasks
from load.loader import save_result
from transform.image.augment import mosaic_transform 
from transform.image.default import mosaic 


def process_augment(args: dict, save_dir: str) -> None:

    target = args
    result = detection_transform(target, augment_transform) # Transform
    save_result(result, save_dir) # Load
    return

def process_copy(args: dict, save_dir: str) -> None:

    target = args
    result = detection_transform(target, copy_transform) # Transform
    save_result(result, save_dir) # Load
    return

def process_mixup(args: list, save_dir: str) -> None:

    target, ex_target = args
    result = mixup_transform(target, ex_target) # Transform
    save_result(result, save_dir) # Load
    return 

def process_mosaic(args: list, save_dir: str) -> None:

    target, mosaic_metadata = args
    result = mosaic_transform(target, mosaic, mosaic_metadata) # Transform
    save_result(result, save_dir) # Load
    return 

# def process_copypaste(args: list, save_dir: str) -> None:

#     target, copypaste_metadata = args
#     result = copypaste_transform(target, copypaste, copypaste_metadata)
#     save_result(result, save_dir)
#     return 

# Pipeline
def detection_pipeline(file_dir: str, 
                       save_dir: str, 
                       aug_mult: int, 
                       copy_mult: bool, 
                       mixup_mult: bool,
                       mosaic_mult: bool) -> None:

    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")

    # Ingest 
    dataset = yolo_dataset(file_dir)

    # Augmentation
    if aug_mult: print("Augmentation Initiated")
    for i in range(aug_mult):
        f_aug = partial(process_augment, save_dir = save_dir)
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f_aug, dataset), total=len(dataset), desc=f"Iter {i+1}"):
                pass

    # Copy 
    if copy_mult: print("Copy Initiated")
    for i in range(copy_mult):
        f_copy = partial(process_copy, save_dir = save_dir)
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f_copy, dataset), total=len(dataset), desc=f"Iter {i+1}"):
                pass

    # Mixup
    if mixup_mult: print("Mixup Augmentation Initiated")
    for i in range(mixup_mult):
        tasks = create_mixup_tasks(dataset)
        f_mixup = partial(process_mixup, save_dir = save_dir)
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f_mixup, tasks), total=len(tasks), desc=f"Iter {i+1}"):
                pass

    # Mosaic
    if mosaic_mult: print("Mosaic Augmentation Initiated")
    for i in range(mosaic_mult):
        tasks = create_mosaic_tasks(dataset)
        f_mosaic = partial(process_mosaic, save_dir = save_dir)
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f_mosaic, tasks), total=len(tasks), desc=f"Iter {i+1}"):
                pass
    

    # # CopyPaste
    # for i in range(copypaste_mult):
    #     tasks = create_copypaste_tasks(dataset)
    #     print("CopyPaste Iteration:", i+1)
    #     f_copypaste = partial(process_mosaic, save_dir = save_dir)
    #     with Pool(processes=cpu_count()) as pool:
    #         for _ in tqdm.tqdm(pool.imap_unordered(f_copypaste, tasks), total=len(tasks), desc="Progress"):
    #             pass
    #     print("Augmentation Completed")

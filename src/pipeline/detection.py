import tqdm
import albumentations as A
from functools import partial
from multiprocessing import Pool, cpu_count
from dataset.yolo import yolo_dataset
from transform.augment import detection_transform 
from transform.custom import MixUp
from utils.helper import create_save_dir, create_tasks 
from extract.extractor import extract_target, extract_mixup_metadata, extract_mosaic_metadata
from load.loader import save_result


# ETL Process
def process_augment(args: dict, 
                    file_dir: str, 
                    save_dir: str, 
                    transform: A.Compose,
                    has_mixup: bool,
                    has_mosaic: bool) -> None:

    file_name = args["album"]
    mosaic_file_name = args["mosaic"]
    mixup_file_name = args["mixup"]

    mixup_metadata = None
    mosaic_metadata = None
    
    # Extract
    target = extract_target(file_dir, file_name)
    if has_mixup:
        mixup_metadata = extract_mixup_metadata(file_dir, mixup_file_name)
    if has_mosaic:
        mosaic_metadata = extract_mosaic_metadata(file_dir, mosaic_file_name)


    if target: 
        # Transform
        transform.set_random_seed(None)
        result = detection_transform(target=target, 
                                     transform=transform,
                                     mixup_metadata=mixup_metadata,
                                     mosaic_metadata=mosaic_metadata)
        # Load
        save_result(result, save_dir)

    return


# Pipeline
def detection_pipeline(file_dir: str, 
                       save_dir: str, 
                       mode: str,
                       transform: A.Compose,
                       mosaic_allocate: int,
                       multiplier: int) -> None:

    create_save_dir(file_dir, save_dir, mode)
    print(f"[\033[32mINFO\033[0m] Mode:            {mode}")
    print(f"[\033[32mINFO\033[0m] Source:          {file_dir}")
    print(f"[\033[32mINFO\033[0m] Destination:     {save_dir}")
    print(f"[\033[32mINFO\033[0m] Augmentation:    {' | '.join(t.__class__.__name__ for t in transform.transforms)}")

    # Dataset 
    dataset = yolo_dataset(file_dir)
    print("YOLO Dataset Created...")

    has_mixup = any(isinstance(t, MixUp) for t in transform.transforms)
    has_mosaic = any(isinstance(t, A.Mosaic) for t in transform.transforms)

    mixup_allocate = 0
    if has_mixup:
        mixup_allocate = 1

    print("Augmentation Initiated...")
    for i in range(multiplier):

        tasks = create_tasks(dataset, 
                             mixup_allocate,
                             mosaic_allocate)

        func = partial(process_augment, 
                       file_dir=file_dir, 
                       save_dir=save_dir,
                       transform=transform,
                       has_mixup=has_mixup,
                       has_mosaic=has_mosaic)

        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(func, tasks), total=len(tasks), desc=f"Iter {i+1}"):
                pass
    print()

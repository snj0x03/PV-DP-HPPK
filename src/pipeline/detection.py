import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from dataset.yolo import yolo_dataset
from transform.image.default import augment_transform, copy_transform 
from transform.image.augment import detection_transform, mixup_transform
from utils.helper import create_save_dir, make_pair_list
from load.loader import save_file_random

def process_target(args: list, copy: bool, multiplier: int, mixup: bool, save_dir: str) -> None:

    target, random_target = args

    if copy:
        result = detection_transform(target, copy_transform) # Transform
        # Load
        if result:
            image, labels, bboxes = result
            try:
                save_file_random(image, labels, bboxes, save_dir) 
            except:
                pass

    for _ in range(multiplier):
        result = detection_transform(target, augment_transform) 
        # Load
        if result:
            image, labels, bboxes = result
            try:
                save_file_random(image, labels, bboxes, save_dir) 
            except:
                pass
            

    if mixup:
        result = mixup_transform(target, random_target) # Transform
        # Load
        if result:
            image, labels, bboxes = result
            try: 
                save_file_random(image, labels, bboxes, save_dir)
            except:
                pass

    return 
    
def detection_pipeline(file_dir: str, save_dir: str, copy: bool, multiplier: int, mixup: bool) -> None:

    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")

    # Ingest 
    dataset = yolo_dataset(file_dir)
    print("Yolo Dataset Created")

    tasks = make_pair_list(dataset)

    print("Augmentation Process Initiated")
    f = partial(process_target, 
                copy = copy, 
                multiplier = multiplier, 
                mixup = mixup, 
                save_dir = save_dir)

    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(f, tasks), total=len(tasks), desc="Progress"):
            pass
    print("Augmentation Completed")


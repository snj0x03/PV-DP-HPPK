from dataset.image import image_dataset
from transform.image.default import classify_transform, empty_transform
from transform.image.augment import classification_transform
from utils.helper import create_save_dir
from multiprocessing import Pool, cpu_count
from functools import partial
from load.loader import save_image_random
import tqdm

def process_target(args: dict, copy: bool, multiplier: int) -> None:

    target = args

    if copy:
        result = classification_transform(target, empty_transform) # Transform
        # Load
        if result: 
            image, image_save_dir = result
            save_image_random(image, image_save_dir) 

    for _ in range(multiplier):
        result = classification_transform(target, classify_transform) # Transform
        # Load
        if result:
            image, image_save_dir = result
            save_image_random(image, image_save_dir) 

    return 

def classification_pipeline(file_dir, save_dir, copy, multiplier):

    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")

    # Ingest
    dataset = image_dataset(file_dir, save_dir)
    print("Image Dataset Created")

    f = partial(process_target, 
                copy = copy, 
                multiplier = multiplier)

    print("Augmentation Process Initiated")
    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(f, dataset), total=len(dataset), desc="Progress"):
            pass

    print("Augmentation Completed")

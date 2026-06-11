import tqdm
from dataset.image import image_dataset
from transform.image.default import classify_transform, empty_transform
from transform.image.augment import classification_transform
from utils.helper import create_save_dir
from multiprocessing import Pool, cpu_count
from functools import partial
from extract.extractor import extract_image
from load.loader import save_image_random

def process_augment(args: dict, file_dir: str, save_dir: str) -> None:

    file_name = args[0]
    label = args[1]
   
    target = extract_image(file_dir, label, file_name) # Extract
    image = classification_transform(target, classify_transform) # Transform 
    save_image_random(image, save_dir, label) # Load
    return

def process_copy(args: dict, file_dir: str, save_dir: str) -> None:

    file_name = args[0]
    label = args[1]
   
    target = extract_image(file_dir, label, file_name) # Extract
    image = classification_transform(target, empty_transform) # Transform 
    save_image_random(image, save_dir, label) # Load
    return

def classification_pipeline(file_dir: str,
                            save_dir: str,
                            mode: str,
                            aug_mult: int,
                            copy_mult: int) -> None:

    create_save_dir(file_dir, save_dir, mode)
    print("Save Directory Created")

    # Dataset
    dataset = image_dataset(file_dir)
    print("Image Dataset Created")

    # Augmentation
    print("Augmentation Initiated")
    for i in range(aug_mult):
        f_aug = partial(process_augment, file_dir = file_dir, save_dir = save_dir)
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f_aug, dataset), total=len(dataset), desc=f"Iter {i+1}"):
                pass
    
    # Copy
    print("Copy Initiated")
    for i in range(copy_mult):
        f_aug = partial(process_augment, file_dir = file_dir, save_dir = save_dir)
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f_aug, dataset), total=len(dataset), desc=f"Iter {i+1}"):
                pass


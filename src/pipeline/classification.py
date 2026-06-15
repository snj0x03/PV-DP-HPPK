import tqdm
import albumentations as A
from functools import partial
from multiprocessing import Pool, cpu_count

from dataset.image import image_dataset
from extract.extractor import extract_image
from transform.augment import classification_transform
from load.loader import save_image_random
from utils.helper import create_save_dir


# ETL Process
def process_augment(args: dict, 
                    file_dir: str, 
                    save_dir: str,
                    transform: A.Compose) -> None:

    file_name = args[0]
    label = args[1]
   
    target = extract_image(file_dir, label, file_name) # Extract
    image = classification_transform(target, transform) # Transform 
    save_image_random(image, save_dir, label) # Load

    return

# Pipeline
def classification_pipeline(file_dir: str,
                            save_dir: str,
                            mode: str,
                            transform: A.Compose,
                            multiplier: int) -> None:

    create_save_dir(file_dir, save_dir, mode)
    print(f"[INFO] Mode:        {mode}")
    print(f"[INFO] Source:      {file_dir}")
    print(f"[INFO] Destination: {save_dir}")

    # Dataset
    dataset = image_dataset(file_dir)
    print("Image Dataset Created...")

    # Augmentation
    print("Augmentation Initiated...")
    for i in range(multiplier):

        func = partial(process_augment, 
                       file_dir=file_dir, 
                       save_dir=save_dir,
                       transform=transform)

        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(func, dataset), total=len(dataset), desc=f"Iter {i+1}"):
                pass
    


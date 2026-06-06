import os
import pandas as pd
import numpy as np

def create_save_dir(file_dir: str, save_dir: str, mode: str) -> None:
    if mode == "normal":
        f_exists = True
    if mode == "strict":
        f_exists = False
    if mode != "normal" or mode != "strict":
        raise ValueError("The mode needs to be either 'normal' or 'strict'")

    for dirname in os.listdir(file_dir):
        if os.path.isdir(os.path.join(file_dir, dirname)):
            os.makedirs(os.path.join(save_dir, dirname), exist_ok=f_exists)

def load_part_names(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df = pd.DataFrame(df)
    part_names = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return part_names 

def create_mosaic_metadata(dataset: list, indexes: list):
    metadata = [
        {
            "image": dataset[i]["image"],
            "bboxes": dataset[i]["bboxes"],
            "bbox_labels": {"labels": dataset[i]["labels"]}
        }
        for i in indexes
    ]
    return metadata

        
def create_mixup_tasks(dataset: list):
    n = len(dataset)
    p1 = np.random.permutation(n)
    p2 = np.random.permutation(n)
    task = [ 
        [dataset[i1], dataset[i2]]
        for i1, i2 in zip(p1, p2)
    ]
    return task


def create_mosaic_tasks(dataset: list):
    n = len(dataset)
    p1 = np.random.permutation(n)
    p2 = np.random.permutation(n)
    p3 = np.random.permutation(n)
    p4 = np.random.permutation(n)
    task = [
        [dataset[i1], create_mosaic_metadata(dataset, [i2, i3, i4])]
        for i1, i2, i3, i4 in zip(p1, p2, p3, p4)
    ]
    return task


# def create_copypaste_metadata(dataset: list):
#     metadata = [
#         {
#             "image": img["image"],
#             "bbox": img["bboxes"][0],
#             "bbox_labels": {"labels": img["labels"][0]}
#         }
#         for img in dataset
#     ]
#     return metadata

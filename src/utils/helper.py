import os
import pandas as pd
import numpy as np

def create_save_dir(file_dir: str, save_dir: str, mode: str) -> None:

    if mode == "normal":
        f_exists = True
    elif mode == "strict":
        f_exists = False
    else:
        raise ValueError("The mode needs to be either 'normal' or 'strict'")

    for dirname in os.listdir(file_dir):
        if os.path.isdir(os.path.join(file_dir, dirname)):
            os.makedirs(os.path.join(save_dir, dirname), exist_ok=f_exists)

def load_part_names(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df = pd.DataFrame(df)
    part_names = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return part_names 

def create_tasks(dataset: list, mixup_allocate: int, mosaic_allocate: int):

    n = len(dataset)
    p_mixup = []
    p_mosaic = []


    if mixup_allocate == 1:
        p_mixup.append(np.random.permutation(n))
    for _ in range(mosaic_allocate):
        p_mosaic.append(np.random.permutation(n))

    tasks = []
    for i in range(n):
        temp_mixup = []
        temp_mosaic = []

        for j in range(mixup_allocate):
            temp_mixup.append(dataset[p_mixup[j][i]])
        for j in range(mosaic_allocate):
            temp_mosaic.append(dataset[p_mosaic[j][i]])

        task = {
            "album": dataset[i],
            "mixup": temp_mixup,
            "mosaic": temp_mosaic
        }
        tasks.append(task)
    return tasks

    
# CopyAndPaste Metadata
# metadata = [
#     {
#         "image": img["image"],
#         "bbox": img["bboxes"][index],
#         "bbox_labels": {"labels": img["labels"][index]}
#     }
# ]

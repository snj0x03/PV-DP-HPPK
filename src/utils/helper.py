import os
import random
import pandas as pd

def create_save_dir(file_dir: str, save_dir: str) -> None:
    for dirname in os.listdir(file_dir):
        if os.path.isdir(os.path.join(file_dir, dirname)):
            os.makedirs(os.path.join(save_dir, dirname))

def load_part_names(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, header=0)
    part_names = {} 
    for _, row in df.iterrows():
        part_names[str(row[0])] = str(row[1])

    return part_names 

def make_pair_list(dataset: list[dict]) -> list[list]:
    total = len(dataset) - 1
    pairs = []
    for x in dataset:
        idx = random.randint(0, total)
        pairs.append([x, dataset[idx]])
    return pairs


# General-purpose filesystem and data-preparation helpers shared across pipelines.

import os
import random
import pandas as pd


def create_save_dir(file_dir: str, save_dir: str) -> None:
    """
    Mirror the immediate subdirectory structure of file_dir under save_dir.
    Creates one output folder per class/part folder found in the source.
    Does not recurse deeper than one level.
    """
    for dirname in os.listdir(file_dir):
        if os.path.isdir(os.path.join(file_dir, dirname)):
            os.makedirs(os.path.join(save_dir, dirname), exist_ok=True)


def load_part_names(csv_path: str) -> dict:
    """
    Read a two-column CSV (no header row assumed in column 0/1) and return
    a dict mapping folder_name → readable part name.

    CSV format example:
        P001,SVC_HP LaserJet Fuser 220V Kit
        P002,SVC_HP LaserJet CYM Managed Imaging Drum
    """
    df = pd.read_csv(csv_path, header=0)
    return {str(row[0]): str(row[1]) for _, row in df.iterrows()}


def make_pair_list(dataset: list[dict]) -> list[list]:
    """
    For each entry in dataset, randomly pair it with another entry (may be itself).
    Used to pre-generate MixUp partners before multiprocessing begins.
    Returns [[target, random_partner], ...], one pair per dataset item.
    """
    total = len(dataset) - 1
    return [[x, dataset[random.randint(0, total)]] for x in dataset]


def make_mosaic_list(dataset: list[dict]) -> list[list]:
    """
    For each entry in dataset, randomly sample 3 additional entries.
    Used to pre-generate Mosaic groups of 4 before multiprocessing begins.
    Returns [[target, r1, r2, r3], ...], one group per dataset item.
    """
    total = len(dataset) - 1
    return [[x] + [dataset[random.randint(0, total)] for _ in range(3)] for x in dataset]

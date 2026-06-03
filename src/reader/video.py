# Scans a root video directory and builds a list of extraction targets.
# Each target maps one .mp4 file to its HP part name via a CSV lookup.

import os
from utils.helpers import load_part_names


def video_dataset(file_dir: str, save_dir: str, csv_path: str) -> list[dict]:
    """
    Walk file_dir, find every .mp4 inside each subfolder, and pair it with
    the corresponding HP part name from the CSV.

    Directory layout expected:
        file_dir/
            P001/  ← folder name used as CSV key
                clip1.mp4
            P002/
                clip2.mp4

    Returns a list of dicts, one per video file:
        {"video_path": ..., "save_dir": ..., "part_name": ...}
    Folders not found in the CSV are silently skipped.
    """
    part_names = load_part_names(csv_path)
    dataset = []

    for dir_name in os.listdir(file_dir):
        sub_dir = os.path.join(file_dir, dir_name)
        if not os.path.isdir(sub_dir):
            continue
        for file_name in os.listdir(sub_dir):
            if file_name.endswith(".mp4"):
                try:
                    target = {
                        "video_path": os.path.join(sub_dir, file_name),
                        "save_dir":   os.path.join(save_dir, dir_name),
                        "part_name":  part_names[dir_name],
                    }
                    dataset.append(target)
                except KeyError:
                    pass  # folder name not in CSV

    return dataset

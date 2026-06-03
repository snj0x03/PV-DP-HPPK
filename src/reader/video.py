# Scans a root video directory and builds a list of extraction targets.
# Classification mode: maps each subfolder name to an HP part name via CSV.
# Detection mode: no CSV needed — frames are saved with UUID filenames only,
#                 since class info will be assigned later during annotation.

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


def video_dataset_detection(file_dir: str, save_dir: str) -> list[dict]:
    """
    Walk file_dir and collect every .mp4 for Detection-mode extraction.
    No CSV mapping is needed — each frame will be saved with a UUID filename
    and annotated with bounding boxes later using an external tool.

    Directory layout expected:
        file_dir/
            scene_01/   ← folder name used only as output subfolder
                clip.mp4
            scene_02/
                clip.mp4

    Returns a list of dicts, one per video file:
        {"video_path": ..., "save_dir": ...}
    """
    dataset = []

    for dir_name in os.listdir(file_dir):
        sub_dir = os.path.join(file_dir, dir_name)
        if not os.path.isdir(sub_dir):
            continue
        for file_name in os.listdir(sub_dir):
            if file_name.endswith(".mp4"):
                dataset.append({
                    "video_path": os.path.join(sub_dir, file_name),
                    "save_dir":   os.path.join(save_dir, dir_name),
                })

    return dataset

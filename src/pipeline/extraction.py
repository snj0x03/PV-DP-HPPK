# Video frame extraction pipeline.
# Distributes extraction across all CPU cores using multiprocessing.
#
# Classification mode: maps subfolder names → HP part names via CSV.
#   Filenames: {part_name}-{uuid}.jpg
#   After extraction, prints class distribution + imbalance warning.
#
# Detection mode: no CSV — frames saved as {uuid}.jpg only.
#   Class info is assigned later during bbox annotation (Roboflow / AnyLabeling).
#   No class distribution report (classes are unknown at this stage).

from functools import partial
from multiprocessing import Pool, cpu_count

import tqdm

from reader.video import video_dataset, video_dataset_detection
from save.writer import save_image_random, save_image_random_part
from augment.video.extract import extract_frame
from utils.helpers import create_save_dir
from utils.stats import count_classes, imbalance_report


def process_target(
    args: dict,
    frame_rate: float,
    deduplicate: bool = False,
    hash_size: int = 8,
    hash_threshold: int = 5,
) -> None:
    """
    Worker for Classification extraction.
    Saves each frame as {part_name}-{uuid}.jpg for traceability.
    """
    video_path, frame_save_dir, part_name = args.values()

    frames = extract_frame(
        video_path, frame_rate,
        deduplicate=deduplicate,
        hash_size=hash_size,
        hash_threshold=hash_threshold,
    )

    for frame in frames:
        save_image_random_part(frame, frame_save_dir, part_name)


def process_target_detection(
    args: dict,
    frame_rate: float,
    deduplicate: bool = False,
    hash_size: int = 8,
    hash_threshold: int = 5,
) -> None:
    """
    Worker for Detection extraction.
    Saves each frame as {uuid}.jpg — no part name embedded, since a single
    frame may contain multiple parts and class info comes from annotation later.
    """
    video_path, frame_save_dir = args.values()

    frames = extract_frame(
        video_path, frame_rate,
        deduplicate=deduplicate,
        hash_size=hash_size,
        hash_threshold=hash_threshold,
    )

    import os
    os.makedirs(frame_save_dir, exist_ok=True)
    for frame in frames:
        save_image_random(frame, frame_save_dir)


def frame_extraction_pipeline(
    file_dir: str,
    save_dir: str,
    frame_rate: float,
    csv_path: str,
    task: str = "Classification",
    deduplicate: bool = False,
    hash_size: int = 8,
    hash_threshold: int = 5,
    warn_ratio: float = 3.0,
) -> None:
    """
    Full extraction pipeline. Behavior differs by task:

    Classification:
        - Requires csv_path to map subfolder names → part names.
        - Frames saved as {part_name}-{uuid}.jpg inside per-folder subdirs.
        - Prints class distribution report after extraction.

    Detection:
        - No CSV needed. Frames saved as {uuid}.jpg inside per-folder subdirs.
        - No class distribution report (classes are unknown until annotation).
        - After extraction, annotate with Roboflow / AnyLabeling to add bboxes.
    """
    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")

    if task == "Detection":
        dataset = video_dataset_detection(file_dir, save_dir)
        print(f"Video Dataset Created ({len(dataset)} video(s)) — Detection mode")

        f = partial(
            process_target_detection,
            frame_rate=frame_rate,
            deduplicate=deduplicate,
            hash_size=hash_size,
            hash_threshold=hash_threshold,
        )

        print("Extraction Process Initiated")
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f, dataset), total=len(dataset), desc="Progress"):
                pass

        print("Frame Extraction Completed")
        print("Next step: annotate extracted frames with bounding boxes (Roboflow / AnyLabeling)")

    else:  # Classification
        dataset = video_dataset(file_dir, save_dir, csv_path)
        print(f"Video Dataset Created ({len(dataset)} video(s)) — Classification mode")

        f = partial(
            process_target,
            frame_rate=frame_rate,
            deduplicate=deduplicate,
            hash_size=hash_size,
            hash_threshold=hash_threshold,
        )

        print("Extraction Process Initiated")
        with Pool(processes=cpu_count()) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(f, dataset), total=len(dataset), desc="Progress"):
                pass

        print("Frame Extraction Completed")
        counts = count_classes(save_dir)
        imbalance_report(counts, warn_ratio=warn_ratio, save_dir=save_dir)

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
) -> int:
    """
    Worker for Classification extraction.
    Saves each frame as {part_name}-{uuid}.jpg for traceability.
    Returns the number of frames saved.
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

    return len(frames)


def process_target_detection(
    args: dict,
    frame_rate: float,
    deduplicate: bool = False,
    hash_size: int = 8,
    hash_threshold: int = 5,
) -> int:
    """
    Worker for Detection extraction.
    Saves each frame as {uuid}.jpg — no part name embedded, since a single
    frame may contain multiple parts and class info comes from annotation later.
    Returns the number of frames saved.
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

    return len(frames)


def frame_extraction_pipeline(
    file_dir: str,
    save_dir: str,
    frame_rate: float,
    csv_path: str,
    video_type: str = "single",
    deduplicate: bool = False,
    hash_size: int = 8,
    hash_threshold: int = 5,
    warn_ratio: float = 3.0,
) -> None:
    """
    Full extraction pipeline. Behavior differs by video_type:

    "single" — one part per frame:
        - Requires csv_path to map subfolder names → part names.
        - Frames saved as {part_name}-{uuid}.jpg inside per-folder subdirs.
        - Prints class distribution report after extraction.

    "multi" — multiple parts per frame:
        - No CSV needed. Frames saved as {uuid}.jpg inside per-folder subdirs.
        - No class distribution report (classes unknown until bbox annotation).
        - After extraction, annotate with Roboflow / AnyLabeling to add bboxes.
    """
    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")

    if video_type == "multi":
        dataset = video_dataset_detection(file_dir, save_dir)
        print(f"Video Dataset Created ({len(dataset)} video(s)) — multi-part mode")

        f = partial(
            process_target_detection,
            frame_rate=frame_rate,
            deduplicate=deduplicate,
            hash_size=hash_size,
            hash_threshold=hash_threshold,
        )

        print("Extraction Process Initiated")
        total_frames = 0
        with Pool(processes=cpu_count()) as pool:
            with tqdm.tqdm(total=len(dataset), desc="Videos") as pbar:
                for frame_count in pool.imap_unordered(f, dataset):
                    total_frames += frame_count
                    pbar.set_postfix(frames=total_frames)
                    pbar.update(1)

        print(f"Frame Extraction Completed — {total_frames} frame(s) saved")
        print("Next step: annotate extracted frames with bounding boxes (Roboflow / AnyLabeling)")

    else:  # single
        dataset = video_dataset(file_dir, save_dir, csv_path)
        print(f"Video Dataset Created ({len(dataset)} video(s)) — single-part mode")

        f = partial(
            process_target,
            frame_rate=frame_rate,
            deduplicate=deduplicate,
            hash_size=hash_size,
            hash_threshold=hash_threshold,
        )

        print("Extraction Process Initiated")
        total_frames = 0
        with Pool(processes=cpu_count()) as pool:
            with tqdm.tqdm(total=len(dataset), desc="Videos") as pbar:
                for frame_count in pool.imap_unordered(f, dataset):
                    total_frames += frame_count
                    pbar.set_postfix(frames=total_frames)
                    pbar.update(1)

        print(f"Frame Extraction Completed — {total_frames} frame(s) saved")
        counts = count_classes(save_dir)
        imbalance_report(counts, warn_ratio=warn_ratio, save_dir=save_dir)

# Video frame extraction pipeline.
# Distributes extraction across all CPU cores using multiprocessing.
# After extraction, prints a class distribution report to flag imbalance early.

from functools import partial
from multiprocessing import Pool, cpu_count

import tqdm

from reader.video import video_dataset
from save.writer import save_image_random_part
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
    Worker function: extract frames from one video and save them to disk.
    Called once per video file inside a multiprocessing Pool.
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


def frame_extraction_pipeline(
    file_dir: str,
    save_dir: str,
    frame_rate: float,
    csv_path: str,
    deduplicate: bool = False,
    hash_size: int = 8,
    hash_threshold: int = 5,
    warn_ratio: float = 3.0,
) -> None:
    """
    Full extraction pipeline:
    1. Mirror source subdirectory structure under save_dir.
    2. Build the list of (video, output_dir, part_name) targets from the CSV.
    3. Distribute extraction across all CPU cores.
    4. Print a class distribution report; warn if imbalance exceeds warn_ratio.
    """
    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")

    dataset = video_dataset(file_dir, save_dir, csv_path)
    print("Video Dataset Created")

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

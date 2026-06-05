# Train/val/test split pipeline.
# Reads a class-per-folder dataset, splits it by the configured strategy,
# and writes the result to save_dir/train|val|test/<class>/.

from utils.split import (
    build_files_by_class, chunk_split, random_split, save_split,
    collect_files_detection, chunk_split_flat, random_split_flat, save_split_detection,
)
from utils.stats import count_classes, imbalance_report, split_distribution_report


def split_pipeline(
    source_dir: str,
    save_dir: str,
    mode: str = "chunk",
    ratios: list = None,
    seed: int = 42,
    chunk_size: int = 10,
    warn_ratio: float = 3.0,
    task: str = "Classification",
) -> None:
    """
    Full split pipeline:

    Classification:
        source_dir/<class>/ → save_dir/train|val|test/<class>/
        Reports class imbalance and per-split distribution.

    Detection:
        source_dir/images/ + labels/ → save_dir/train|val|test/images/ + labels/
        Images and their label files are split together (no class-folder structure).

    chunk mode (recommended for video-sourced data):
        Groups consecutive frames into chunks, shuffles whole chunks to prevent
        near-duplicate frames from leaking across train/val/test boundaries.

    random mode:
        Shuffles individual files — simpler but may leak near-duplicates.
    """
    if ratios is None:
        ratios = [0.7, 0.15, 0.15]

    if task == "Detection":
        files = collect_files_detection(source_dir)
        print(f"Detection split: {len(files)} image(s) found")

        if mode == "chunk":
            print(f"Split mode: chunk (chunk_size={chunk_size}, seed={seed})")
            splits_flat = chunk_split_flat(files, ratios, chunk_size, seed)
        else:
            print(f"Split mode: random (seed={seed})")
            splits_flat = random_split_flat(files, ratios, seed)

        for split_name, split_files in splits_flat.items():
            print(f"  {split_name}: {len(split_files)} image(s)")

        save_split_detection(splits_flat, save_dir, source_dir)
        print(f"\nDetection split saved to: {save_dir}")
        return

    # Classification
    counts = count_classes(source_dir)
    imbalance_report(counts, warn_ratio=warn_ratio, save_dir=save_dir)

    files_by_class = build_files_by_class(source_dir)

    if mode == "chunk":
        print(f"\nSplit mode: chunk (chunk_size={chunk_size}, seed={seed})")
        splits = chunk_split(files_by_class, ratios, chunk_size, seed)
    else:
        print(f"\nSplit mode: random (seed={seed})")
        splits = random_split(files_by_class, ratios, seed)

    split_counts = {
        split_name: {cls: len(files) for cls, files in class_dict.items()}
        for split_name, class_dict in splits.items()
    }
    split_distribution_report(split_counts, save_dir=save_dir)

    save_split(splits, save_dir)
    print(f"\nSplit saved to: {save_dir}")

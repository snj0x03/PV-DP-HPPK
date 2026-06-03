# Train/val/test split pipeline.
# Reads a class-per-folder dataset, splits it by the configured strategy,
# and writes the result to save_dir/train|val|test/<class>/.

from utils.split import build_files_by_class, chunk_split, random_split, save_split
from utils.stats import count_classes, imbalance_report, split_distribution_report


def split_pipeline(
    source_dir: str,
    save_dir: str,
    mode: str = "chunk",
    ratios: list = None,
    seed: int = 42,
    chunk_size: int = 10,
    warn_ratio: float = 3.0,
) -> None:
    """
    Full split pipeline:
    1. Report class distribution and warn if imbalance exceeds warn_ratio.
    2. Build per-class file lists from source_dir.
    3. Split using 'chunk' or 'random' mode.
    4. Print the per-class split distribution table.
    5. Copy files to save_dir/train|val|test/<class>/.

    chunk mode (recommended for video-sourced data):
        Groups consecutive frames into chunks of chunk_size, then shuffles
        and assigns whole chunks to splits. Prevents near-duplicate frames
        from leaking across the train/val/test boundary.

    random mode:
        Shuffles individual files per class — simpler but may leak near-duplicates.
    """
    if ratios is None:
        ratios = [0.7, 0.15, 0.15]

    counts = count_classes(source_dir)
    imbalance_report(counts, warn_ratio=warn_ratio, save_dir=save_dir)

    files_by_class = build_files_by_class(source_dir)

    if mode == "chunk":
        print(f"\nSplit mode: chunk (chunk_size={chunk_size}, seed={seed})")
        splits = chunk_split(files_by_class, ratios, chunk_size, seed)
    else:
        print(f"\nSplit mode: random (seed={seed})")
        splits = random_split(files_by_class, ratios, seed)

    # Convert path lists to counts for the distribution report
    split_counts = {
        split_name: {cls: len(files) for cls, files in class_dict.items()}
        for split_name, class_dict in splits.items()
    }
    split_distribution_report(split_counts, save_dir=save_dir)

    save_split(splits, save_dir)
    print(f"\nSplit saved to: {save_dir}")

# Entry point for the PVision data pipeline.
# Loads sys_config.yml, parses the --option argument, and dispatches to the
# appropriate pipeline function.
#
# Available options:
#   extract  — extract frames from raw .mp4 videos
#   augment  — augment an annotated dataset (Classification or Detection)
#   split    — split an image dataset into train/val/test
#   lc       — create staged subsets for learning curve experiments

import os
import warnings
import argparse
import yaml

from pipeline.extraction import frame_extraction_pipeline
from pipeline.classification import classification_pipeline
from pipeline.detection import detection_pipeline


def _check_save_dir(save_dir: str, option: str) -> None:
    """Warn the user if save_dir already contains files and ask for confirmation.
    Exits early if the user declines, preventing accidental data accumulation."""
    if not os.path.isdir(save_dir):
        return
    existing = [
        f for root, _, files in os.walk(save_dir)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".txt"))
    ]
    if not existing:
        return
    print(f"\n[WARNING] save_dir already contains {len(existing)} file(s):")
    print(f"  → {save_dir}")
    print(f"  New files from '{option}' will be ADDED on top of existing files.")
    print(f"  To start fresh, use a different save_dir path in sys_config.yml.")
    answer = input("Proceed? [y/N]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        raise SystemExit(0)


def main():
    parser = argparse.ArgumentParser(description="PVision Data Pipeline")
    parser.add_argument("-o", "--option", type=str,
                        choices=["extract", "augment", "split", "lc"],
                        help="Pipeline stage to run")
    args = parser.parse_args()

    with open("./conf/sys_config.yml", "r") as f:
        CFG = yaml.safe_load(f)

    # --- Extraction ---
    VIDEO_DIR      = CFG["video_dir"]
    FRAME_SAVE_DIR = CFG["frame_save_dir"]
    FRAME_RATE     = CFG["frame_rate"]
    CSV_PATH       = CFG["csv_path"]

    # --- Augmentation ---
    YOLO_DIR      = CFG["yolo_dir"]
    YOLO_SAVE_DIR = CFG["yolo_save_dir"]
    COPY_ORIGINAL = CFG["copy"]
    MULTIPLIER    = CFG["multiplier"]
    MIXUP         = CFG["mixup"]
    MOSAIC        = CFG.get("mosaic", False)
    TASK          = CFG["task"]

    # --- Deduplication (extraction stage) ---
    DEDUPLICATE    = CFG.get("deduplicate", False)
    HASH_SIZE      = CFG.get("hash_size", 8)
    HASH_THRESHOLD = CFG.get("hash_threshold", 5)

    # --- Split ---
    SPLIT_MODE   = CFG.get("split_mode", "chunk")
    SPLIT_RATIOS = CFG.get("split_ratios", [0.7, 0.15, 0.15])
    SPLIT_SEED   = CFG.get("split_seed", 42)
    CHUNK_SIZE   = CFG.get("chunk_size", 10)

    # --- Shared ---
    IMBALANCE_WARN = CFG.get("imbalance_ratio_warn", 3.0)

    # --- Learning curve ---
    LC_STAGES = CFG.get("lc_stages", [50, 100, 150, 300])
    LC_SEED   = CFG.get("lc_seed", 42)

    if args.option == "extract":
        _check_save_dir(FRAME_SAVE_DIR, "extract")
        frame_extraction_pipeline(
            VIDEO_DIR, FRAME_SAVE_DIR, FRAME_RATE, CSV_PATH,
            task=TASK,
            deduplicate=DEDUPLICATE,
            hash_size=HASH_SIZE,
            hash_threshold=HASH_THRESHOLD,
            warn_ratio=IMBALANCE_WARN,
        )

    if args.option == "augment" and TASK == "Detection":
        _check_save_dir(YOLO_SAVE_DIR, "augment")
        detection_pipeline(YOLO_DIR, YOLO_SAVE_DIR, COPY_ORIGINAL, MULTIPLIER, MIXUP, MOSAIC)
        from utils.stats import count_classes, imbalance_report
        imbalance_report(count_classes(YOLO_SAVE_DIR), warn_ratio=IMBALANCE_WARN, save_dir=YOLO_SAVE_DIR)

    if args.option == "augment" and TASK == "Classification":
        _check_save_dir(YOLO_SAVE_DIR, "augment")
        classification_pipeline(YOLO_DIR, YOLO_SAVE_DIR, COPY_ORIGINAL, MULTIPLIER)
        from utils.stats import count_classes, imbalance_report
        imbalance_report(count_classes(YOLO_SAVE_DIR), warn_ratio=IMBALANCE_WARN, save_dir=YOLO_SAVE_DIR)

    if args.option == "split":
        _check_save_dir(YOLO_SAVE_DIR, "split")
        from pipeline.split import split_pipeline
        split_pipeline(
            YOLO_DIR, YOLO_SAVE_DIR,
            mode=SPLIT_MODE,
            ratios=SPLIT_RATIOS,
            seed=SPLIT_SEED,
            chunk_size=CHUNK_SIZE,
            warn_ratio=IMBALANCE_WARN,
        )

    if args.option == "lc":
        _check_save_dir(YOLO_SAVE_DIR, "lc")
        from pipeline.learning_curve import learning_curve_pipeline
        learning_curve_pipeline(
            YOLO_DIR, YOLO_SAVE_DIR,
            stages=LC_STAGES,
            seed=LC_SEED,
            warn_ratio=IMBALANCE_WARN,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

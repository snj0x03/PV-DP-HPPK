# Learning curve dataset generator.
# Creates a series of progressively larger dataset subsets (stages) from a
# source directory so that model performance can be measured at each data volume.
# Each stage is a random sample of min(N, available) images per class.

import os
import random
import shutil

from utils.stats import count_classes, imbalance_report


def learning_curve_pipeline(
    source_dir: str,
    save_dir: str,
    stages: list = None,
    seed: int = 42,
    warn_ratio: float = 3.0,
) -> None:
    """
    For each stage N in stages, sample N images per class from source_dir and
    copy them to save_dir/stage_N/<class>/.

    Sampling is seed-controlled (random.Random, not global random) so each
    stage is independently reproducible without affecting the rest of the program.
    If a class has fewer than N images, all of its images are included.

    After creating each stage, prints a class distribution report so you can
    check imbalance at every data volume level.

    Typical usage:
        After running this pipeline, train a model on each stage directory and
        record validation accuracy. Plot accuracy vs. images-per-class to find
        the point of diminishing returns — this is the "learning curve".
    """
    if stages is None:
        stages = [50, 100, 150, 300]

    # Collect and sort files per class for reproducibility
    files_by_class = {}
    for class_name in sorted(os.listdir(source_dir)):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        files = sorted(
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if files:
            files_by_class[class_name] = files

    for stage in stages:
        stage_dir = os.path.join(save_dir, f"stage_{stage}")
        # Use a fresh RNG instance per stage so stages are independent
        rng = random.Random(seed)

        for cls, files in files_by_class.items():
            sampled  = rng.sample(files, min(stage, len(files)))
            dest_dir = os.path.join(stage_dir, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for src in sampled:
                shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))

        print(f"\n[Learning Curve] stage_{stage}")
        counts = count_classes(stage_dir)
        imbalance_report(counts, warn_ratio=warn_ratio, save_dir=stage_dir)

    print(f"\nLearning curve datasets saved to: {save_dir}")

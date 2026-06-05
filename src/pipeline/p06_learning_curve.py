# Learning curve dataset generator.
# Creates a series of progressively larger dataset subsets (stages) from a
# source directory so that model performance can be measured at each data volume.
# Each stage is a random sample of min(N, available) images per class.

import os
import random
import shutil

from utils.stats import count_classes, imbalance_report
from utils.split import collect_files_detection


def learning_curve_pipeline(
    source_dir: str,
    save_dir: str,
    stages: list = None,
    seed: int = 42,
    warn_ratio: float = 3.0,
    task: str = "Classification",
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

    if task == "Detection":
        _lc_detection(source_dir, save_dir, stages, seed)
        return

    # Classification: sample N images per class at each stage
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


def _lc_detection(source_dir: str, save_dir: str, stages: list, seed: int) -> None:
    """
    Detection-mode learning curve: sample N image+label pairs per stage.
    Output: save_dir/stage_N/images/ + stage_N/labels/
    """
    labels_dir = os.path.join(source_dir, "labels")
    files = collect_files_detection(source_dir)
    print(f"Detection LC: {len(files)} image(s) found")

    for stage in stages:
        stage_dir = os.path.join(save_dir, f"stage_{stage}")
        img_dest  = os.path.join(stage_dir, "images")
        lbl_dest  = os.path.join(stage_dir, "labels")
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        rng     = random.Random(seed)
        sampled = rng.sample(files, min(stage, len(files)))

        for src_img in sampled:
            shutil.copy2(src_img, os.path.join(img_dest, os.path.basename(src_img)))
            stem      = os.path.splitext(os.path.basename(src_img))[0]
            label_src = os.path.join(labels_dir, stem + ".txt")
            if os.path.isfile(label_src):
                shutil.copy2(label_src, os.path.join(lbl_dest, stem + ".txt"))

        print(f"\n[Learning Curve] stage_{stage}: {len(sampled)} image(s)")

    print(f"\nLearning curve datasets saved to: {save_dir}")

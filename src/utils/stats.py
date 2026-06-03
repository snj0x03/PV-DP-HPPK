# Class distribution statistics and imbalance reporting.
# Reports are printed to the terminal and optionally saved as CSV files
# so results can be referenced in the final project report.

import csv
import os


def count_classes(dataset_dir: str) -> dict:
    """
    Count images in each class subfolder under dataset_dir.
    Returns {class_name: image_count} for Classification datasets.
    Non-directory entries are ignored.
    """
    counts = {}
    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        counts[class_name] = sum(
            1 for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    return counts


def count_classes_detection(dataset_dir: str) -> dict:
    """
    Count bbox occurrences per class from YOLO label files in dataset_dir/labels/.

    Each line in a .txt file: class_id cx cy w h
    Returns {"class_0": bbox_count, "class_1": bbox_count, ...} sorted by class id.

    Use this instead of count_classes() for Detection datasets — those use an
    images/+labels/ folder structure, not a class-per-subfolder structure.
    """
    labels_dir = os.path.join(dataset_dir, "labels")
    if not os.path.isdir(labels_dir):
        print(f"[stats] No 'labels/' folder found in: {dataset_dir}")
        return {}

    counts: dict = {}
    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, fname), "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                key = f"class_{int(parts[0])}"
                counts[key] = counts.get(key, 0) + 1

    return dict(sorted(counts.items()))


def imbalance_report(
    class_counts: dict,
    warn_ratio: float = 3.0,
    save_dir: str = None,
) -> None:
    """
    Print a class distribution table and flag potential imbalance.

    Prints each class and its image count (sorted descending), then the
    max:min ratio. Emits a WARNING line if the ratio exceeds warn_ratio
    (default 3.0, meaning the largest class has 3× the smallest).

    If save_dir is given, also writes class_distribution.csv there with
    columns: class, count, ratio_to_min.
    """
    if not class_counts:
        print("[stats] No classes found.")
        return

    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    max_count     = sorted_counts[0][1]
    min_count     = sorted_counts[-1][1]

    print("\n--- Class Distribution ---")
    for cls, cnt in sorted_counts:
        marker = "  ← EMPTY" if cnt == 0 else ""
        print(f"  {cls:<40}: {cnt}{marker}")

    empty_classes = [cls for cls, cnt in sorted_counts if cnt == 0]
    if empty_classes:
        print(f"  WARNING: {len(empty_classes)} class(es) with 0 images: {', '.join(empty_classes)}")
        print(f"  (Max:Min ratio skipped — remove or populate empty classes before training)")
    else:
        ratio = max_count / min_count
        print(f"  Max:Min ratio: {ratio:.2f}")
        if ratio > warn_ratio:
            print(f"  WARNING: ratio {ratio:.2f} exceeds threshold ({warn_ratio})!")
        else:
            print(f"  OK: ratio within threshold ({warn_ratio})")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "class_distribution.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "count", "ratio_to_min"])
            for cls, cnt in sorted_counts:
                if min_count == 0:
                    r = "0" if cnt == 0 else "inf"
                else:
                    r = f"{cnt / min_count:.2f}"
                writer.writerow([cls, cnt, r])
        print(f"  Saved: {csv_path}")


def split_distribution_report(
    splits: dict,
    save_dir: str = None,
) -> None:
    """
    Print a table showing how many images from each class landed in each split.

    splits format: {"train": {class: count}, "val": {class: count}, "test": {class: count}}

    If save_dir is given, also writes split_distribution.csv there.
    """
    all_classes = sorted({cls for split_dict in splits.values() for cls in split_dict})
    split_names = list(splits.keys())

    print("\n--- Split Distribution ---")
    header = f"  {'class':<40}" + "".join(f"{s:>8}" for s in split_names)
    print(header)
    print("  " + "-" * (40 + 8 * len(split_names)))
    for cls in all_classes:
        row = f"  {cls:<40}" + "".join(f"{splits[s].get(cls, 0):>8}" for s in split_names)
        print(row)
    print("  " + "-" * (40 + 8 * len(split_names)))
    total = f"  {'TOTAL':<40}" + "".join(f"{sum(splits[s].values()):>8}" for s in split_names)
    print(total)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "split_distribution.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class"] + split_names)
            for cls in all_classes:
                writer.writerow([cls] + [splits[s].get(cls, 0) for s in split_names])
        print(f"  Saved: {csv_path}")

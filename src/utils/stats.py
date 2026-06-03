# Class distribution statistics and imbalance reporting.
# Reports are printed to the terminal and optionally saved as CSV files
# so results can be referenced in the final project report.

import csv
import os


def count_classes(dataset_dir: str) -> dict:
    """
    Count the number of images (.jpg/.jpeg/.png) in each class subfolder
    under dataset_dir.  Returns {class_name: count}, sorted by class name.
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
    ratio         = (max_count / min_count) if min_count > 0 else float("inf")

    print("\n--- Class Distribution ---")
    for cls, cnt in sorted_counts:
        print(f"  {cls:<40}: {cnt}")
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
                r = f"{cnt / min_count:.2f}" if min_count > 0 else "inf"
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

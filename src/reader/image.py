# Scans a classification image directory and builds a list of augmentation targets.
# Expects one subfolder per class, each containing .jpg images.

import os


def image_dataset(file_dir: str, save_dir: str) -> list[dict]:
    """
    Walk file_dir, treating each subfolder as a class label.
    Returns one dict per image found:
        {"image_path": ..., "save_dir": ...}

    save_dir is the matching output folder so the augmented image
    lands in the correct class subfolder.

    Directory layout expected:
        file_dir/
            class_A/  img1.jpg  img2.jpg
            class_B/  img1.jpg
    """
    dataset = []

    for class_name in os.listdir(file_dir):
        class_dir = os.path.join(file_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            if file_name.endswith(".jpg"):
                try:
                    target = {
                        "image_path": os.path.join(class_dir, file_name),
                        "save_dir":   os.path.join(save_dir, class_name),
                    }
                    dataset.append(target)
                except Exception:
                    pass

    return dataset

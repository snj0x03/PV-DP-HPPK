import os

def yolo_dataset(file_dir: str) -> list[dict]:
    image_dir = os.path.join(file_dir, "images")
    label_dir = os.path.join(file_dir, "labels")
    # List of file names
    dataset = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith((".jpg")):
            try:
                if label_dir:
                    dataset.append(file_name)
            except:
                pass

    return dataset

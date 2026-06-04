import os
import cv2

def image_dataset(file_dir: str) -> list[dict]:
    dataset = []
    for dir_name in os.listdir(file_dir):
        part = dir_name
        new_dir_name = os.path.join(file_dir, dir_name)
        if os.path.isdir(new_dir_name):
            for file_name in os.listdir(new_dir_name):
                if file_name.endswith((".jpg")):
                    try:
                        image_path = os.path.join(new_dir_name, file_name)
                        image = cv2.imread(image_path)
                        target = {
                            "image": image,
                            "label": part
                        }
                        dataset.append(target)
                    except:
                        pass
    return dataset

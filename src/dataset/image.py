import os

def image_dataset(file_dir: str, save_dir:str) -> list[dict]:
    dataset = []
    for dir_name in os.listdir(file_dir):
        part = dir_name
        new_dir_name = os.path.join(file_dir, dir_name)
        if os.path.isdir(new_dir_name):
            for file_name in os.listdir(new_dir_name):
                if file_name.endswith((".jpg")):
                    target = {
                        "image_path": os.path.join(new_dir_name, file_name),
                        "save_dir": os.path.join(save_dir, part)
                    }
                    dataset.append(target)
    return dataset

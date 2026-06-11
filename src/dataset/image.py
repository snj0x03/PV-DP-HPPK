import os

def image_dataset(file_dir: str) -> list[dict]:
    # List of [image_path, class]
    dataset = []
    for dir_name in os.listdir(file_dir):
        part = dir_name
        new_dir_name = os.path.join(file_dir, dir_name)
        if os.path.isdir(new_dir_name):
            for file_name in os.listdir(new_dir_name):
                if file_name.endswith((".jpg")):
                    try:
                        target = [
                            file_name,
                            part
                        ]
                        dataset.append(target)
                    except:
                        pass
    return dataset

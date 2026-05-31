import os
from utils.helper import load_part_names

def video_dataset(file_dir: str, save_dir: str, csv_path: str) -> list[dict]:
    part_names = load_part_names(csv_path)
    dataset = []
    for dir_name in os.listdir(file_dir):
        part = dir_name
        new_dir_name = os.path.join(file_dir, dir_name)
        if os.path.isdir(new_dir_name):
            for file_name in os.listdir(new_dir_name):
                if file_name.endswith((".mp4")):
                    try:
                        target =  {
                            "video_path": os.path.join(new_dir_name, file_name), 
                            "save_dir": os.path.join(save_dir, part), 
                            "part_name": part_names[part]
                        }
                        dataset.append(target)
                    except:
                        pass
    return dataset 


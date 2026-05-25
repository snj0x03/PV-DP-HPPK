from utils.directory import load_part_names
import os

def video_dataset(file_dir: str, save_dir: str, csv_path: str) -> list[dict[str, str]]:
    part_names = load_part_names(csv_path)
    dataset = []
    for directory in os.listdir(file_dir):
        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith((".mp4")):
                    target =  {
                        "video_path": os.path.join(directory, filename), 
                        "save_dir": os.path.join(save_dir, directory), 
                        "part_name": part_names[directory]
                    }
                    dataset.append(target)

    return dataset 

from utils.directory import load_part_names
import os


def video_dataset(file_dir: str, save_dir: str, csv_path: str) -> list[dict[str, str]]:
    part_names = load_part_names(csv_path)
    dataset = []
    for dirname in os.listdir(file_dir):
        p = dirname
        d = os.path.join(file_dir, dirname)
        if os.path.isdir(d):
            for filename in os.listdir(d):
                if filename.endswith((".mp4")):
                    target =  {
                        "video_path": os.path.join(d, filename), 
                        "save_dir": os.path.join(save_dir, p), 
                        "part_name": part_names[p]
                    }
                    dataset.append(target)

    return dataset 

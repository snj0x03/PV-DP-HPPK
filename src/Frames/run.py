from utils.directory import create_save_dir
from Dataset.video import video_dataset
from Frames.transform import extract_frame


def run_extract(main_dir: str, save_dir: str, frame_rate: int, csv_path: str) -> None:

    create_save_dir(main_dir, save_dir)
    print("Save Directory Created")

    dataset = video_dataset(main_dir, save_dir, csv_path)
    print("Video Dataset Created")

    cnt = 0
    for target in dataset:
        video_path = target["video_path"]
        frame_save_dir = target["frame_save_dir"]
        part_name = target["part_name"]
        extract_frame(video_path, frame_save_dir, frame_rate, part_name)
        cnt += 1
        print(f"Progress - {cnt / len(dataset)}%", end="\r", flush=True)
    print("Frame Extraction Completed")

    

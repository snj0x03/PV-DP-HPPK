from utils.directory import setup_save_dir
from Dataset.video import video_dataset
from Frames.transform import extract_frame


def run_extract(video_dir: str, frame_save_dir: str, frame_rate: int, csv_path: str) -> None:

    setup_save_dir(video_dir, frame_save_dir)

    dataset = video_dataset(video_dir, frame_save_dir, csv_path)

    for target in dataset:
        video_dir = target["video_path"]
        frame_save_dir = target["frame_save_dir"]
        part_name = target["part_name"]
        extract_frame(video_dir, frame_save_dir, frame_rate, part_name)

    

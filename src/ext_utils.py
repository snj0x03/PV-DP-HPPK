# Utils 

import cv2
import os 
import uuid
import pandas as pd 
# from PIL import Image


def setup_save_dir(video_dir: str, frame_save_dir: str) -> None:
    for directory in os.listdir(video_dir):
        if os.path.isdir(directory):
            os.makedirs(os.path.join(frame_save_dir, directory))



def load_part_names(csv_path: str) -> dict[str, str]:
    df = pd.read_csv(csv_path, header=0)
    part_names = {} 
    for _, row in df.iterrows():
        part_names[str(row[0])] = str(row[1])

    return part_names 
        
            
    
def video_dataset(video_dir: str, frame_save_dir: str, csv_path: str) -> list[dict[str, str]]:
    part_names = load_part_names(csv_path)
    dataset = []
    for directory in os.listdir(video_dir):
        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith((".mp4")):
                    target =  {
                        "video_path": os.path.join(directory, filename), 
                        "frame_save_dir": os.path.join(frame_save_dir, directory), 
                        "part_name": part_names[directory]
                    }
                    dataset.append(target)

    return dataset 
    


def extract_frame(video_dir: str, frame_save_dir: str, frame_rate: int, part_name: str) -> None:
    cap = cv2.VideoCapture(video_dir)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(int(fps * frame_rate), 1)

    count = 0
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % interval == 0:
            try:
                # img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                filename = f"{part_name}-{uuid.uuid1()}.jpg"
                filepath = os.path.join(frame_save_dir, filename)
                cv2.imwrite(filepath, frame)
            except Exception as e:
                print(e)
    
    return 


def run_extract(video_dir: str, frame_save_dir: str, frame_rate: int, csv_path: str) -> None:

    setup_save_dir(video_dir, frame_save_dir)

    dataset = video_dataset(video_dir, frame_save_dir, csv_path)

    for target in dataset:
        video_dir = target["video_path"]
        frame_save_dir = target["frame_save_dir"]
        part_name = target["part_name"]
        extract_frame(video_dir, frame_save_dir, frame_rate, part_name)

    

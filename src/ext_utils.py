# Utils 

import os 


def setup_save_dir(video_dir, frame_save_dir):
    for dir_name in video_dir:
        os.makedirs(frame_save_dir, dir_name)


def video_list(video_dir, frame_save_dir, csv_path):
    csv = load(csv_path)
    v_list = []
    for folder in video_dir:
        for video in folder:
            target =  {"video_path": folder+video, "save_dir": frame_save_dir+folder, "part_name": csv[folder]}
            v_list.append(target)

    return v_list 
    


def extract_frame(video_dir, frame_save_dir, frame_rate, part_name):
    


def run_extract(video_dir, frame_save_dir, frame_rate):

    setup_save_dir(video_dir, frame_save_dir)

    for target in video_list(video_dir, frame_save_dir, csv_path):
        video_dir = target["video_path"]
        frame_save_dir = target["save_dir"]
        part_name = target["part_name"]
        extract_frame(video_dir, frame_save_dir, frame_rate, part_name)

    

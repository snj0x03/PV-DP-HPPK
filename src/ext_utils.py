# Utils 

import os 

def video_folder_list 


def match_csv_folder(folder_list: , part_list)


def setup_extract_dir(extract_folder: str):
    for dir_name in folder_list:
        os.makedirs(base, dir_name)


def extract_list(extract_folder):
    for folder in folder_list:
        for video in folder:
            target =  {"vp": folder+video, "dp": extract_folder+folder}

    


def extract_frame(video_path, save_dir, frame_rate, part_name):
    


def run_extract(extract_list, extract_folder):
    setup_extract_dir(extract_folder)
    for target in extract_list(extract_folder):
        video_path = target["vid path"]
        dst = target["dst"]
        extract_frame(video_path, destination_folder, frame_rate)

    

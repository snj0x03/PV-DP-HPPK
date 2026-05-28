import yaml
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "./src")

from Frames.run import run_extract 
from Augmentation.run import run_augment, run_augment2

# MAIN 
def main():

    # ARGUMENT PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str)
    args   = parser.parse_args()

    # LOAD YAML
    with open("config.yml", "r") as f:
        CFG = yaml.safe_load(f)
    
    # ARGUMENTS 
    VIDEO_DIR = CFG["video_dir"]
    FRAME_SAVE_DIR = CFG["frame_save_dir"]
    FRAME_RATE = CFG["frame_rate"]
    CSV_PATH = CFG["csv_path"]

    YOLO_DIR = CFG["yolo_dir"]
    YOLO_SAVE_DIR = CFG["yolo_save_dir"]
    COPY_ORIGINAL = CFG["copy"]
    MULTIPLIER = CFG["multiplier"]
    MIXUP = CFG["mixup"]
    TASK = CFG["task"]


    # RUN VIDEO FRAME EXTRACTION
    if args.option == "extract":
        run_extract(VIDEO_DIR, FRAME_SAVE_DIR, FRAME_RATE, CSV_PATH)


    # RUN AUGMENTATION 
    if args.option == "augment" and TASK == "Detection":
        run_augment(YOLO_DIR, YOLO_SAVE_DIR, COPY_ORIGINAL, MULTIPLIER, MIXUP)

    if args.option == "augment" and TASK == "Classification":
        run_augment2(YOLO_DIR, YOLO_SAVE_DIR, COPY_ORIGINAL, MULTIPLIER)



if __name__ == "__main__":
    main()

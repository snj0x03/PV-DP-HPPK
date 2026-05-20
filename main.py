import yaml
import os
import sys
import argparse

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from run_extraction  import run_extract
from run_augmentation import run_augment

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

    AUGMENTATION_LIST = [
        CFG["hflip"],
        CFG["rflip"],
        CFG["rotate"],
        CFG["blur"],
        CFG["noise"],
        CFG["mixup"],
        CFG["mosaic"],
    ]


    # RUN VIDEO FRAME EXTRACTION
    if args.option == "extract":
        run_extract(VIDEO_DIR, FRAME_SAVE_DIR, FRAME_RATE, CSV_PATH)


    # RUN AUGMENTATION 
    if args.option == "augment":
        run_augment(YOLO_DIR, YOLO_SAVE_DIR, AUGMENTATION_LIST)


if __name__ == "__main__":
    main()

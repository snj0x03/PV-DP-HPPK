import yaml
import argparse
import warnings
from pipeline.extraction import frame_extraction_pipeline 
from pipeline.classification import classification_pipeline
from pipeline.detection import detection_pipeline 

def main():
    # Parse Argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str)
    args   = parser.parse_args()

    # Load Config
    with open("./conf/sys_config.yml", "r") as f:
        CFG = yaml.safe_load(f)

    # Extraction Config
    VIDEO_DIR = CFG["video_dir"]
    FRAME_SAVE_DIR = CFG["frame_save_dir"]
    FRAME_RATE = CFG["frame_rate"]
    CSV_PATH = CFG["csv_path"]

    # Augmentation Config
    YOLO_DIR = CFG["yolo_dir"]
    YOLO_SAVE_DIR = CFG["yolo_save_dir"]
    COPY_ORIGINAL = CFG["copy"]
    MULTIPLIER = CFG["multiplier"]
    MIXUP = CFG["mixup"]
    TASK = CFG["task"]

    # run frame extraction on video data
    if args.option == "extract":
        frame_extraction_pipeline(VIDEO_DIR, FRAME_SAVE_DIR, FRAME_RATE, CSV_PATH)

    # run augmentation on yolo data
    if args.option == "augment" and TASK == "Detection":
        detection_pipeline(YOLO_DIR, YOLO_SAVE_DIR, COPY_ORIGINAL, MULTIPLIER, MIXUP)

    # run augmentation on classification data
    if args.option == "augment" and TASK == "Classification":
        classification_pipeline(YOLO_DIR, YOLO_SAVE_DIR, COPY_ORIGINAL, MULTIPLIER)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

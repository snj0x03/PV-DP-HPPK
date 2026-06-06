import yaml
import argparse
import warnings
import cv2
import time
from pipeline.extraction import frame_extraction_pipeline 
from pipeline.classification import classification_pipeline
from pipeline.detection import detection_pipeline 

def main():
    # Optimization
    cv2.setNumThreads(0)

    # Parse Argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str)
    args   = parser.parse_args()

    # Load Config
    with open("./conf/sys_config.yml", "r") as f:
        CFG = yaml.safe_load(f)

    
    # Mode
    MODE = CFG["mode"]

    # Extraction Config
    VIDEO_DIR = CFG["video_dir"]
    FRAME_SAVE_DIR = CFG["frame_save_dir"]
    FRAME_RATE = CFG["frame_rate"]
    CSV_PATH = CFG["csv_path"]

    # Augmentation Config
    IMAGE_DIR = CFG["image_dir"]
    IMAGE_SAVE_DIR = CFG["image_save_dir"]
    TASK = CFG["task"]

    COPY_MULT = CFG["copy_mult"]
    AUG_MULT = CFG["aug_mult"]
    MIXUP_MULT = CFG["mixup_mult"]
    MOSAIC_MULT = CFG["mosaic_mult"]
    
    start = time.time()

    # run frame extraction on video data
    if args.option == "extract":
        frame_extraction_pipeline(file_dir=VIDEO_DIR,
                                  save_dir=FRAME_SAVE_DIR, 
                                  mode = MODE,
                                  frame_rate=FRAME_RATE, 
                                  csv_path=CSV_PATH)

    # run augmentation on yolo data
    if args.option == "augment" and TASK == "Detection":
        detection_pipeline(file_dir=IMAGE_DIR, 
                           save_dir=IMAGE_SAVE_DIR, 
                           mode = MODE,
                           aug_mult=AUG_MULT,
                           copy_mult=COPY_MULT, 
                           mixup_mult=MIXUP_MULT,
                           mosaic_mult=MOSAIC_MULT)

    # run augmentation on classification data
    if args.option == "augment" and TASK == "Classification":
        classification_pipeline(file_dir=IMAGE_DIR, 
                                save_dir=IMAGE_SAVE_DIR,
                                mode = MODE,
                                copy_mult=COPY_MULT,
                                aug_mult=AUG_MULT)

    end = time.time()
    print("Processing time:", end - start)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

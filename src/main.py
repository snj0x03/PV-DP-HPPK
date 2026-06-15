import yaml
import argparse
import warnings
import cv2
import time
import conf.cls_config as cls_flow 
import conf.det_config as det_flow
from pipeline.extraction import frame_extraction_pipeline 
from flow.orchestrate import classification_orchestrate, detection_orchestrate

def main():
    # Optimization
    cv2.setNumThreads(0)

    # Parse Argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str)
    args   = parser.parse_args()

    # Load Frame Extraction Config
    with open("./conf/ext_config.yml", "r") as f:
        CFG = yaml.safe_load(f)

    # Mode
    MODE = CFG["mode"]

    # Extraction Config
    VIDEO_DIR = CFG["video_dir"]
    FRAME_SAVE_DIR = CFG["frame_save_dir"]
    FRAME_RATE = CFG["frame_rate"]
    CSV_PATH = CFG["csv_path"]


    start = time.time()

    # run frame extraction on video data
    if args.option == "ext":
        frame_extraction_pipeline(file_dir=VIDEO_DIR,
                                  save_dir=FRAME_SAVE_DIR, 
                                  mode = MODE,
                                  frame_rate=FRAME_RATE, 
                                  csv_path=CSV_PATH)

    # run augmentation on yolo data
    if args.option == "cls":
        classification_orchestrate(AUG_LIST=cls_flow.AUG_LIST, 
                                   MODE=cls_flow.MODE)

    # run augmentation on classification data
    if args.option == "det":
        detection_orchestrate(AUG_LIST=det_flow.AUG_LIST, 
                              MODE=det_flow.MODE)

    end = time.time()
    print("Processing time:", end - start)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

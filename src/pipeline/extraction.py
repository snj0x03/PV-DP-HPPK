from utils.helper import create_save_dir
from dataset.video import video_dataset
from load.loader import save_image_random_part
from functools import partial
from multiprocessing import Pool, cpu_count
import tqdm
import cv2

def etl_frame(video_path: str, 
              save_dir: str, 
              frame_rate: float, 
              part: str, 
              label: str,
              index: int) -> None:

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(int(fps * frame_rate), 1)

    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            try:
                save_image_random_part(frame, save_dir, part, label, index)
            except:
                pass
        count += 1
    
    cap.release()

def process_target(args: dict, frame_rate: float, save_dir: str) -> None:

    video_path = args[0] 
    label = args[1] 
    part = args[2] 
    index = args[3]

    etl_frame(video_path, save_dir, frame_rate, part, label, index) # ETL
    
    return 

def frame_extraction_pipeline(file_dir: str, 
                              save_dir: str, 
                              mode: str,
                              frame_rate: float, 
                              csv_path: str) -> None:

    print(f"[\033[32mINFO\033[0m] Mode:            {mode}")
    print(f"[\033[32mINFO\033[0m] Source:          {file_dir}")
    print(f"[\033[32mINFO\033[0m] Destination:     {save_dir}")
    print(f"[\033[32mINFO\033[0m] Frame Rate:      {frame_rate}")

    create_save_dir(file_dir, save_dir, mode)
    print("Save Directory Created...")
   
    # Dataset 
    dataset = video_dataset(file_dir, csv_path)
    print("Video Dataset Created...")

    print("Extraction Process Initiated...")
    f = partial(process_target, 
                frame_rate = frame_rate,
                save_dir = save_dir)

    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(f, dataset), total=len(dataset), desc="Progress"):
            pass
    print()




    

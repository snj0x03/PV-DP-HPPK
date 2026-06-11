from utils.helper import create_save_dir
from dataset.video import video_dataset
from transform.video.extract import extract_frame
from load.loader import save_image_random_part
from functools import partial
from multiprocessing import Pool, cpu_count
import tqdm

def process_target(args: dict, frame_rate: float, save_dir: str) -> None:

    video_path, label, part = args.values()

    frame_list = extract_frame(video_path, frame_rate) # Transform
    
    for frame in frame_list:
        save_image_random_part(frame, save_dir, part, label) # Load

    return 

def frame_extraction_pipeline(file_dir: str, 
                              save_dir: str, 
                              mode: str,
                              frame_rate: float, 
                              csv_path: str) -> None:

    create_save_dir(file_dir, save_dir, mode)
    print("Save Directory Created")
   
    # Dataset 
    dataset = video_dataset(file_dir, csv_path)
    print("Video Dataset Created")

    print("Extraction Process Initiated")
    f = partial(process_target, 
                frame_rate = frame_rate,
                save_dir = save_dir)

    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(f, dataset), total=len(dataset), desc="Progress"):
            pass




    

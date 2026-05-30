from utils.helper import create_save_dir
from dataset.video import video_dataset
from transform.video.extract import extract_frame
from load.loader import save_image_random_part
from functools import partial
from multiprocessing import Pool, cpu_count
import tqdm

def process_target(args: dict, frame_rate: float) -> None:

    video_path, frame_save_dir, part_name = args.values()

    frame_list = extract_frame(video_path, frame_rate) # Transform
    
    for frame in frame_list:
        save_image_random_part(frame, frame_save_dir, part_name) # Load

    return 

def frame_extraction_pipeline(file_dir: str, save_dir: str, frame_rate: float, csv_path: str) -> None:

    create_save_dir(file_dir, save_dir)
    print("Save Directory Created")
   
    # Ingest 
    dataset = video_dataset(file_dir, save_dir, csv_path)
    print("Video Dataset Created")

    print("Extraction Process Initiated")
    f = partial(process_target, 
                frame_rate = frame_rate)

    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(f, dataset), total=len(dataset), desc="Progress"):
            pass

    print("Frame Extraction Completed")



    

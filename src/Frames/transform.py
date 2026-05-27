import cv2
import uuid
from utils.directory import save_image

def extract_frame(video_path: str, save_dir: str, frame_rate: int, part_name: str) -> None:
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
                filename = f"{part_name}-{uuid.uuid1()}.jpg"
                save_image(frame, save_dir, filename)
            except Exception as e:
                print(e)
        count += 1
    
    return 

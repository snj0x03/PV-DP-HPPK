import cv2
import uuid
import os

def extract_frame(video_dir: str, frame_save_dir: str, frame_rate: int, part_name: str) -> None:
    cap = cv2.VideoCapture(video_dir)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(int(fps * frame_rate), 1)

    count = 0
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % interval == 0:
            try:
                # img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                filename = f"{part_name}-{uuid.uuid1()}.jpg"
                filepath = os.path.join(frame_save_dir, filename)
                cv2.imwrite(filepath, frame)
            except Exception as e:
                print(e)
    
    return 
import cv2

def extract_frame(video_path: str, frame_rate: float) -> list:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(int(fps * frame_rate), 1)

    count = 0
    frame_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(frame)
            except Exception as e:
                print(e)
        count += 1
    
    cap.release()
    return frame_list

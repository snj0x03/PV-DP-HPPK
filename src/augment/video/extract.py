# Extracts frames from a single video file at a fixed time interval.
# Optionally filters near-duplicate frames using average-hash deduplication.

import cv2
from utils.dedup import filter_duplicates


def extract_frame(
    video_path: str,
    frame_rate: float,
    deduplicate: bool = False,
    hash_size: int = 8,
    hash_threshold: int = 5,
) -> list:
    """
    Sample one frame every frame_rate seconds from the video at video_path.

    frame_rate is converted to a frame-skip interval using the video's actual FPS:
        interval = max(int(fps * frame_rate), 1)
    e.g. frame_rate=0.8 on a 30fps video → extract every 24th frame (~1.25 fps).

    If deduplicate=True, near-duplicate frames are removed using average-hash
    comparison within the extracted list (Hamming distance ≤ hash_threshold).

    Returns a list of RGB numpy arrays, one per kept frame.
    """
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
                # OpenCV reads BGR; convert to RGB for consistency with PIL/albumentations
                frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(e)
        count += 1

    cap.release()

    if deduplicate:
        frame_list = filter_duplicates(frame_list, hash_size, hash_threshold)

    return frame_list

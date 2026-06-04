import os
import cv2

def yolo_dataset(file_dir: str) -> list[dict]:
    image_dir = os.path.join(file_dir, "images")
    label_dir = os.path.join(file_dir, "labels")
    dataset = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith((".jpg")):
            try:
                image_path = os.path.join(image_dir, file_name)
                label_path = os.path.join(label_dir, file_name.replace(".jpg", ".txt"))

                image = cv2.imread(image_path)

                labels, bboxes = [], []
                with open(label_path, "r") as f:
                    for lines in f.readlines():
                        label, boxes = lines.split()[0], lines.split()[1:]
                        label = int(label)
                        boxes = [float(box) for box in boxes]
                        labels.append(label)
                        bboxes.append(boxes)

                target = {
                    "image": image, 
                    "labels": labels,
                    "bboxes": bboxes 
                }

                dataset.append(target)
            except:
                pass

    return dataset

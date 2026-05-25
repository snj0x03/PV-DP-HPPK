import os

def yolo_dataset(file_dir: str) -> list[dict[str, str | list[int] | list[list[int]]]]:
    image_dir = os.path.join(file_dir, "images")
    label_dir = os.path.join(file_dir, "labels")
    dataset = []
    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg")):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

            labels, bboxes = [], []
            with open(label_path, "r") as f:
                for lines in f.readlines():
                    label, boxes = lines.split()[0], lines.split()[1:]
                    label = int(label)
                    boxes = [float(box) for box in boxes]
                    labels.append(label)
                    bboxes.append(boxes)

            target = {
                "image_path": image_path, 
                "labels": labels,
                "bboxes": bboxes 
            }

            dataset.append(target)

    return dataset

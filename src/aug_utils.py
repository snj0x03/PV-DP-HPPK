# Utils 

import os

# apply hflip
def horizontal_flip(target, yolo_save_dir):
    return 

# apply vflip
def vertical_flip(target, yolo_save_dir):
    return 

# apply blur 
def blur(target, yolo_save_dir):
    return 

# apply noise
def noise(target, yolo_save_dir):
    return 

# apply rotate
def rotate(target, yolo_save_dir):
    return 

# apply random scale 
def random_scale(target, yolo_save_dir):
    return 

# apply Mosaic
def mosaic(target, yolo_save_dir):
    return 

# apply mixup
def mixup(target, yolo_save_dir):
    return



def yolo_dataset(yolo_dir: str) -> list[dict[str, str | list[int] | list[list[int]]]]:
    image_dir = os.path.join(yolo_dir, "images")
    label_dir = os.path.join(yolo_dir, "labels")
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




def run_augment(yolo_dir: str, yolo_save_dir: str, augmentation_list: dict[str, dict[str, str | int]]) -> None:

    dataset = yolo_dataset(yolo_dir)

    for target in dataset:
        if augmentation_list["horizontal_flip"]["apply"] == True:
            horizontal_flip(target, yolo_save_dir)

        if augmentation_list["vertical_flip"]["apply"] == True:
            vertical_flip(target, yolo_save_dir)
        
        if augmentation_list["blur"]["apply"] == True:
            blur(target, yolo_save_dir)

        if augmentation_list["noise"]["apply"] == True:
            noise(target, yolo_save_dir)

        if augmentation_list["rotate"]["apply"] == True:
            rotate(target, yolo_save_dir)

        if augmentation_list["mosaic"]["apply"] == True:
            mosaic(target, yolo_save_dir)

        if augmentation_list["mixup"]["apply"] == True:
            mixup(target, yolo_save_dir)
            

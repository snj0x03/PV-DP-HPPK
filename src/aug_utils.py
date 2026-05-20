# Utils 

import os

# apply hflip
def horizontal_flip(target, save_dir):

# apply vflip
def vertical_flip(target, save_dir):

# apply blur 
def blur(target, save_dir):

# apply noise
def noise(target, save_dir):

# apply rotate
def rotate(target, save_dir):

# apply random scale 
def random_scale(target, save_dir):

# apply Mosaic
def mosaic(target, save_dir):

# apply mixup
def mixup(target, save_dir):

# dataset
def yolo_dataset(yolo_dir):
    for image in yolo_dir/images:
        target = {path: "image_path", bboxes: [], labels: []}
        dataset.append(target)

    return dataset

def apply_augmentation(yolo_dir, yolo_save_dir, augmentation_list):

    dataset = yolo_dataset(yolo_dir)

    for target in dataset:

        if augmentation_list["hflip"]["apply"] == True:
            apply_hflip(target, yolo_save_dir)

        if augmentation_list["vflip"]["apply"] == True:
            apply_hflip(target, yolo_save_dir)
        
        if augmentation_list["blur"]["apply"] == True:
            apply_hflip(target, yolo_save_dir)

        if augmentation_list["noise"]["apply"] == True:
            apply_hflip(target, yolo_save_dir)

        if augmentation_list["rotate"]["apply"] == True:
            apply_hflip(target, yolo_save_dir)

        if augmentation_list["mosaic"]["apply"] == True:
            apply_hflip(target, yolo_save_dir)

        if augmentation_list["mixup"]["apply"] == True:
            apply_hflip(target, yolo_save_dir)
            

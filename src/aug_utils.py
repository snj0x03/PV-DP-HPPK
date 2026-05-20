# Utils 

import os

from albucore import hflip, vflip
from albumentations import Mosaic
from cv2 import blur
from cv2.detail import Blender_MULTI_BAND 


# apply hflip
# apply vflip
# apply blur 
# apply noise
# apply rotate
# apply random scale 
# apply Mosaic
# apply mixup

# dataset
def yolodataset():
    for image in image_path:
        target = {path: "image_path", bboxes: [], labels: []}
        dataset.append(target)

def apply_all():
    for image in dataset:

        if apply["hflip"] == true:
            apply_hflip(target, save_dir)

        if apply["vflip"] == true:
            apply_hflip(target, save_dir)

        if apply["blur"] == true:
            apply_blur(target, save_dir)

        if apply["noise"] == true:
            apply_noise(target, save_dir)

        if apply["random scale"] == true:
            apply_rscale(target, save_dir)

        if apply["mosaic"] == true:
            apply_mosaic(target, save_dir)

        if apply["mixup"] == true:
            apply_mixup(target, save_dir)
        




            

import albumentations as A
from Augmentation.Custom import *

pipeline_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.GaussNoise(p=0.5),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["labels"],
        min_visibility=0.3,
    ),
)

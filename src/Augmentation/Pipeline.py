import albumentations as A

augment_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.6),
        A.RandomBrightnessContrast(p=0.6),
        A.Rotate(limit=(-30, 30), p=0.6),
        A.GaussianBlur(p=0.5),
        A.GaussNoise(p=0.5),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["labels"],
        min_visibility=0.3,
    ),
)

copy_transform = A.Compose([], bbox_params=A.BboxParams(
    format="yolo",
    label_fields=["labels"],
    min_visibility=0.3,
))

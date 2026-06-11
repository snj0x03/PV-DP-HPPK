import albumentations as A

augment_transform = A.Compose(
    [
        A.SquareSymmetry(p=1.0),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(angle_range=(-60, 60),
                 rotate_method="ellipse",
                 p=1.0),
        A.BBoxSafeRandomCrop(p=0.4),
        A.OneOf([
            A.GaussianBlur(p=1.0),
            A.GaussNoise(p=1.0)
        ], p=0.6),
        A.CoarseDropout(p=0.4),
        A.LetterBox(size=(640, 640), p=1.0)
    ],
    bbox_params=A.BboxParams(
        coord_format="yolo",
        label_fields=["labels"],
        min_visibility=0.3,
    )
)

geometric = A.Compose(
    [
        A.Rotate(angle_range=(-30, 30),
                 rotate_method="ellipse",
                 p=1.0),
    ],
    bbox_params=A.BboxParams(
        coord_format="yolo",
        label_fields=["labels"],
        min_visibility=0.3,
    )
)

copy_transform = A.Compose([
    A.NoOp(p=1.0)
    ], 
    bbox_params=A.BboxParams(
        coord_format="yolo",
        label_fields=["labels"],
        min_visibility=0.3,
))

classify_transform = A.Compose([
    A.HorizontalFlip(p=0.6),
])

empty_transform = A.Compose([
    A.NoOp(p=1.0)
])

mosaic = A.Compose(
    [
        A.CLAHE(p=0.3),
        A.CoarseDropout(p=0.3),
        A.Mosaic(
            grid_yx=(2, 2),
            target_size=(640, 640),
            cell_shape=(320, 320),
            fit_mode="contain",
            p = 1.0
        ),
        A.SquareSymmetry(p=0.8),
    ],
    bbox_params=A.BboxParams(
        coord_format="yolo", 
        label_fields=["labels"],
        min_visibility=0.3
    )
)




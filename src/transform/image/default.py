import albumentations as A

augment_transform = A.Compose(
    [
        A.Affine(
            scale=(0.8, 1.2),
            rotate=(-60, 60),
            translate_px=(30, 30),
            shear={"x": (-10, 10), "y": (-5, 5)},
            p=1.0
        ),
        A.SquareSymmetry(p=1.0),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.4),
    ],
    bbox_params=A.BboxParams(
        coord_format="yolo",
        label_fields=["labels"],
        min_visibility=0.3,
    )
)

geometric = A.Compose(
    [
        A.Affine(
            scale=(0.8, 1.2),
            rotate=(-60, 60),
            translate_px=(30, 30),
            shear={"x": (-10, 10), "y": (-5, 5)},
            p=1.0
        ),
        A.SquareSymmetry(p=1.0),
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
        A.Mosaic(
            grid_yx=(2, 2),
            target_size=(640, 640),
            cell_shape=(320, 320),
            p = 1.0
        ),
        A.SquareSymmetry(p=0.5),
    ],
    bbox_params=A.BboxParams(
        coord_format="yolo", 
        label_fields=["labels"],
        min_visibility=0.3
    )
)

copypaste = A.Compose([
    A.CopyAndPaste(
        min_visibility_after_paste = 0.05,
        scale_range=(0.5, 1.0),
        blend_mode="gaussian",
        blend_sigma_range=(0.2, 0.3),
        p = 1.0
    )],
    bbox_params=A.BboxParams(
        coord_format="yolo", 
        label_fields=["labels"],
        min_visibility=0.3
    )
)




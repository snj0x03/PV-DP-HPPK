import albumentations as A
from transform.custom import MixUp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


MODE = "normal"

Input = "C:\\Users\\sawna\\Desktop\\AUG_TRAIN\\Input"
F1 = "C:\\Users\\sawna\\Desktop\\AUG_TRAIN\\F1"
F2 = "C:\\Users\\sawna\\Desktop\\AUG_TRAIN\\F2"
Output = "C:\\Users\\sawna\\Desktop\\AUG_TRAIN\\Output"

AUG_LIST = [
    {
        "source": Input,
        "destination": F1,
        "multiplier": 2,
        "mosaic_allocate": 0,
        "transform": A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.BBoxSafeRandomCrop(p=0.5),
            A.Affine(rotate=(-30, 30),
                     rotate_method="ellipse",
                     p=1.0),
            A.OneOf([
                A.GaussianBlur(p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.5),
            A.SquareSymmetry(p=1.0),
        ],
        bbox_params=A.BboxParams(
            coord_format="yolo",
            label_fields=["labels"],
            min_visibility=0.3),
        )
    },
    {
        "source": F1,
        "destination": Output,
        "multiplier": 1,
        "mosaic_allocate": 3,
        "transform": A.Compose([
            A.Mosaic(
                grid_yx=(2, 2),
                target_size=(640, 640),
                cell_shape=(320, 320),
                fit_mode="contain",
                p=1.0
            )
        ],
        bbox_params=A.BboxParams(
            coord_format="yolo",
            label_fields=["labels"],
            min_visibility=0.3)
        )
    },
    {
        "source": Input,
        "destination": F2,
        "multiplier": 2,
        "mosaic_allocate": 0,
        "transform": A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-30, 30),
                     rotate_method="ellipse",
                     p=1.0),
            A.BBoxSafeRandomCrop(p=0.5),
            A.LetterBox(size=(640, 640), p=1.0)
        ],
        bbox_params=A.BboxParams(
            coord_format="yolo",
            label_fields=["labels"],
            min_visibility=0.3)
        )
    },
    {
        "source": F2,
        "destination": Output,
        "multiplier": 1,
        "mosaic_allocate": 0,
        "transform": A.Compose([
            MixUp(p=1.0)
        ],
        bbox_params=A.BboxParams(
            coord_format="yolo",
            label_fields=["labels"],
            min_visibility=0.3)
        )
    },
    {
        "source": Input,
        "destination": Output,
        "multiplier": 2,
        "mosaic_allocate": 1,
        "transform": A.Compose([
            A.Affine(rotate=(-30, 30),
                     rotate_method="ellipse",
                     p=0.5),
            A.CLAHE(p=0.3),
            A.Mosaic(
                grid_yx=(2, 1),
                target_size=(640, 640),
                cell_shape=(320, 640),
                fit_mode="contain",
                p=1.0
            )
        ],
        bbox_params=A.BboxParams(
            coord_format="yolo",
            label_fields=["labels"],
            min_visibility=0.3)
        )
    },
    {
        "source": Input,
        "destination": Output,
        "multiplier": 2,
        "mosaic_allocate": 0,
        "transform": A.Compose([
            A.Affine(rotate=(-30, 30),
                     rotate_method="ellipse",
                     p=0.5),
            A.HueSaturationValue(hue_shift_range=(0, 0),
                                 sat_shift_range=(-15, 15),
                                 val_shift_range=(-15, 15),
                                 p=0.5),
            A.GridDropout(p=0.3),
            A.LetterBox(size=(640, 640), p=1.0)
        ],
        bbox_params=A.BboxParams(
            coord_format="yolo",
            label_fields=["labels"],
            min_visibility=0.3)
        )
    },
]


import albumentations as A
from transform.custom import MixUp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


MODE = "normal"

AUG_LIST = [
    {
        "source": "C:\\Users\\sawna\\Desktop\\demo\\P1",
        "destination": "C:\\Users\\sawna\\Desktop\\demo\\PO",
        "multiplier": 1,
        "mosaic_allocate": 0,
        "transform": A.Compose([
            A.LetterBox(size=(640, 640), p=1.0)
        ],
        bbox_params=A.BboxParams(
            coord_format="yolo",
            label_fields=["labels"],
            min_visibility=0.3)
        )
    },
    {
        "source": "C:\\Users\\sawna\\Desktop\\demo\\PO",
        "destination": "C:\\Users\\sawna\\Desktop\\demo\\PO2",
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
]


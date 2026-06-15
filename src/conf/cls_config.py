import albumentations as A
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


MODE = "normal"

AUG_LIST = [
    {
        "source": "C:\\Users\\sawna\\Desktop\\demo\\P3",
        "destination": "C:\\Users\\sawna\\Desktop\\demo\\PO",
        "multiplier": 2,
        "transform": A.Compose([
            A.AdditiveNoise(p=1.0)
        ])
    },
    {
        "source": "C:\\Users\\sawna\\Desktop\\demo\\P3",
        "destination": "C:\\Users\\sawna\\Desktop\\demo\\PO",
        "multiplier": 2,
        "transform": A.Compose([
            A.GridDropout(p=1.0)
        ])
    },
]


import albumentations as A
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


MODE = "normal"

Input = "C:\\Users\\sawna\\Desktop\\tput\\valid"
Output = "C:\\Users\\sawna\\Desktop\\new_cls\\valid"

AUG_LIST = [
    {
        "source": Input,
        "destination": Output,
        "multiplier": 5,
        "transform": A.Compose([
            A.SquareSymmetry(p=0.8),
            A.Affine(
                scale=(1.0, 1.3),
                rotate=(-45, 45),
                shear={"x": (-10, 10), "y":(-5, 5)},
                p=1.0),
            A.HueSaturationValue(hue_shift_range=(0, 0),
                                 sat_shift_range=(-5, 5),
                                 val_shift_range=(-20, 20),
                                 p=1.0),
            A.CLAHE(p=0.1),
            A.Blur(p=0.3),
            A.LetterBox(size=(640, 640), p=1.0)
        ])
    },
    {
        "source": Input,
        "destination": Output,
        "multiplier": 1,
        "transform": A.Compose([
            A.LetterBox(size=(640, 640), p=1.0)
        ])
    },
]


# Pre-defined Albumentations transform pipelines used by the augmentation stage.
# Detection transforms carry BboxParams so bounding boxes are updated in sync with the image.
# Classification transforms operate on images only (no bbox handling needed).

import albumentations as A

# --- Detection transforms ---

# Full augmentation for detection: geometric + color + noise.
# min_visibility=0.3 drops any bbox that becomes < 30% visible after a crop/rotation.
augment_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.6),
        A.RandomBrightnessContrast(p=0.6),
        A.Rotate(limit=(-60, 60), p=0.6),
        A.GaussianBlur(p=0.5),
        A.GaussNoise(p=0.5),
        A.HueSaturationValue(p=0.7),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["labels"],
        min_visibility=0.3,
    ),
)

# No-op detection transform — passes image and bboxes through unchanged.
# Used when copy=True to include the original alongside its augmented variants.
copy_transform = A.Compose(
    [A.NoOp(p=1.0)],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["labels"],
        min_visibility=0.3,
    ),
)

# --- Classification transforms ---

# Augmentation for classification: lighter rotation (±30°) vs detection (±60°)
# because isolated parts are more sensitive to extreme angles.
classify_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.6),
        A.RandomBrightnessContrast(p=0.6),
        A.Rotate(limit=(-30, 30), p=0.6),
        A.GaussianBlur(p=0.5),
        A.GaussNoise(p=0.5),
    ]
)

# No-op classification transform — passes image through unchanged.
empty_transform = A.Compose([A.NoOp(p=1.0)])

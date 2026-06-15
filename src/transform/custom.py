from albumentations import DualTransform
from albumentations.core.bbox_utils import convert_bboxes_to_albumentations
import numpy as np
from typing import Any


class MixUp(DualTransform):
    def __init__(self, metadata_key: str = "mixup_metadata", p: float = 0.5):
        super().__init__(p=p)
        self.metadata_key = metadata_key

    @property
    def targets_as_params(self):
        return [self.metadata_key]

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        external_target = data[self.metadata_key]
        w1, h1 = data["image"].shape[:2]
        w2, h2 = external_target[0]["image"].shape[:2]

        arr1 = external_target[0]["bboxes"]
        arr2 = external_target[0]["bbox_labels"]["labels"]

        bboxes_xywh = np.array(arr1)
        bboxes_xyxy = convert_bboxes_to_albumentations(
            bboxes_xywh,
            source_format='yolo',
            shape=(w2, h2), 
            bbox_type='hbb',
        )

        res = np.column_stack((bboxes_xyxy, arr2))

        return {
          "w1": w1,
          "h1": h1,
          "w2": w2,
          "h2": h2,
          "ex_image": external_target[0]["image"],
          "ex_bboxes": res
        }


    def apply(self, img: np.ndarray, 
              ex_image: np.ndarray, 
              w1: float, 
              h1: float, 
              w2: float, 
              h2: float, 
              **params: Any) -> np.ndarray:
        # w1, h1 = img.shape[:2]
        # w2, h2 = ex_image.shape[:2]
        H, W = max(h1, h2), max(w1, w2)

        c1 = np.pad(img, ((0, int(W - w1)), (0, int(H - h1)), (0, 0)), 'constant')
        c2 = np.pad(ex_image, ((0, int(W - w2)), (0, int(H - h2)), (0, 0)), 'constant')
        lam = np.random.uniform(0.6, 0.7)
        mixed = np.clip(lam * c1 + (1 - lam) * c2, 0, 255).astype(np.uint8)

        return mixed 

    def apply_to_bboxes(self, 
                        bboxes: np.ndarray, 
                        ex_bboxes: np.ndarray, 
                        h1: float, 
                        w1: float, 
                        h2: float, 
                        w2: float, 
                        *args: Any, 
                        **params: Any) -> np.ndarray:

        H, W = max(h1, h2), max(w1, w2)
        for bbox in bboxes:
            bbox[0] = bbox[0] * (h1 / H)
            bbox[1] = bbox[1] * (w1 / W)
            bbox[2] = bbox[2] * (h1 / H)
            bbox[3] = bbox[3] * (w1 / W)
  
        for bbox in ex_bboxes:
            bbox[0] = bbox[0] * (h2 / H)
            bbox[1] = bbox[1] * (w2 / W)
            bbox[2] = bbox[2] * (h2 / H)
            bbox[3] = bbox[3] * (w2 / W)
          

        bboxes = np.vstack((bboxes, ex_bboxes))
        return bboxes 


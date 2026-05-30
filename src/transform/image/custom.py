import numpy as np

def mixup(image: np.ndarray, bboxes: list, labels: list, 
          ex_image: np.ndarray, ex_bboxes: list, ex_labels:list) -> dict:

    w1, h1 = image.shape[:2]
    w2, h2 = ex_image.shape[:2]
    H, W = max(h1, h2), max(w1, w2)

    c1 = np.pad(image, ((0, W - w1), (0, H - h1), (0, 0)), 'constant')
    c2 = np.pad(ex_image, ((0, W - w2), (0, H - h2), (0, 0)), 'constant')

    bboxes = [list(bbox) for bbox in bboxes]
    ex_bboxes = [list(bbox) for bbox in ex_bboxes]

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

    lam = np.random.uniform(0.6, 0.7)
    mixed = np.clip(lam * c1 + (1 - lam) * c2, 0, 255).astype(np.uint8)

    mixed_bboxes = bboxes + ex_bboxes
    mixed_labels = labels + ex_labels 
    result = { "image": mixed, "bboxes": mixed_bboxes, "labels": mixed_labels }

    return result

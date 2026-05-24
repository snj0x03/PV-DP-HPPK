import numpy as np
from PIL import Image

def MixUp(image1, image2):
    
    img1 = Image.open("image1.jpg")
    img2 = Image.open("image2.jpg")

    h1, w1 = img1.size
    h2, w2 = img2.size

    img1 = np.array(img1)
    img2 = np.array(img2)

    H = max(h1, h2)
    W = max(w1, w2)

    c1 = np.pad(img1, ((0, W - w1), (0, H - h1), (0, 0)), 'constant')
    c2 = np.pad(img2, ((0, W - w2), (0, H - h2), (0, 0)), 'constant')

    for bbox in bboxes1:
        bbox[0] = bbox[0] * (h1 / H)
        bbox[1] = bbox[1] * (w1 / W)
        bbox[2] = bbox[2] * (h1 / H)
        bbox[3] = bbox[3] * (w1 / W)

    for bbox in bboxes2:
        bbox[0] = bbox[0] * (h1 / H)
        bbox[1] = bbox[1] * (w1 / W)
        bbox[2] = bbox[2] * (h1 / H)
        bbox[3] = bbox[3] * (w1 / W)

    lam = np.random.uniform(0.6, 0.7)
    mixed = np.clip(lam * c1 + (1 - lam) * c2, 0, 255).astype(np.uint8)

    bboxes = bboxes1 + bbboxes2
    labels = labels1 + labels2
    result = { "img": mixed, "bboxes": bboxes, "labels": labels }

     return result

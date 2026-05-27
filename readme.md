# PVision Data Pipeline

A data preparation pipeline for HP printer part image datasets.  
Handles frame extraction from raw videos and image augmentation for model training.


## Project Structure

```
PV-DP-HPPK/
├── config.yml
├── main.py
├── requirements.txt
└── src/
    ├── run_extraction.py
    ├── run_augmentation.py
    ├── augment.py
    └── utils.py
```


## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd PV-DP-HPPK
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yml` before running:

```yaml
video_dir      : ""      # path to video files
frame_save_dir : ""      # path for output of frame extraction
csv_path       : ""      # path to parts name csv

frame_rate     : 0.25    # seconds per frame (0.25 = 4 fps)
target_max     : 400     # max frames to extract per part

yolo_dir       : ""      # Input YOLO directory
yolo_save_dir  : ""      # Output YOLO directory

blur:
    apply:     : true    # apply blur

noise:
    apply      : true    # apply noise
```


## Usage

Run Video Frame Extraction:

```bash
python main.py --option extract
```

Run augmentation on YOLO dataset:

```bash
python main.py --option augment
```


## Pipeline Stages

### Stage 1 — Frame Extraction
- Reads videos 
- Maps part folders to HP part names via the Excel file
- Extracts frames at 4 fps, up to `target_max` per part
- Output 

### Stage 2 - Annotation
- Annotate Extracte frame with Roboflow or AnyLabeling

### Stage 3 — Augmentation
- Reads annotated images (YOLO format)
- Applies flip, brightness, blur, noise, rotate, MixUp, Mosaic
- Output 



## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Video reading and frame extraction |
| `Pillow` | Image loading and saving |
| `numpy` | Array operations for augmentation |
| `pandas` | Reading the HP parts Excel file |
| `pyyaml` | Parsing `config.yml` |
| `imagehash` | Perceptual hash deduplication |
| `albumentations` | Image augmentation pipeline |

# PVision Data Pipeline

A data preparation pipeline for HP printer part image datasets.  
Handles frame extraction from raw videos and image augmentation for model training.


## Project Structure

```
PV-DP-HPPK-v0/
├── config.yml
├── main.py
├── requirements.txt
└── src/
    ├── Augmentation/
    │   ├── Custom.py
    │   ├── Pipeline.py
    │   ├── run.py
    │   └── transform.py
    ├── Dataset/
    │   ├── video.py
    │   └── yolo.py
    ├── Frames/
    │   ├── run.py
    │   └── transform.py
    └── utils/
        └── directory.py
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
# Frame extraction
video_dir      : ""      # path to raw video folders
frame_save_dir : ""      # path for extracted frame output
csv_path       : ""      # path to parts name CSV
frame_rate     : 45      # seconds per frame (e.g. 0.25 = 4 fps, 45 = 1 frame per 45s)

# Augmentation
yolo_dir       : ""      # input YOLO dataset directory
yolo_save_dir  : ""      # output directory for augmented dataset
task           : "Classification"  # "Detection" or "Classification"
copy           : true    # copy originals into output
multiplier     : 6       # number of augmented copies per image
mixup          : true    # enable MixUp blending (Detection only)
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
| `pandas` | Reading the HP parts CSV file |
| `pyyaml` | Parsing `config.yml` |
| `albumentations` | Image augmentation pipeline |

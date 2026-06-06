# PVision Data Pipeline

A data preparation pipeline for HP printer part image datasets.  
It handles two independent stages: **frame extraction** from raw `.mp4` videos, and **image augmentation** on annotated YOLO-format datasets.

---

## Project Structure

```
PV-DP-HPPK/
├── requirements.txt
└── src/
    ├── main.py                  # Entry point — parse args, load config, run pipeline
    ├── conf/
    │   └── sys_config.yml       # All runtime configuration lives here
    ├── dataset/
    │   ├── image.py             # Scans image folders for classification data
    │   ├── video.py             # Scans video folders, maps folder names to HP part names
    │   └── yolo.py              # Reads YOLO-format images and label .txt files
    ├── pipeline/
    │   ├── extraction.py        # Orchestrates video → frame extraction
    │   ├── detection.py         # Orchestrates YOLO detection augmentation
    │   └── classification.py    # Orchestrates classification augmentation
    ├── transform/
    │   ├── image/
    │   │   ├── default.py       # Albumentations transform presets
    │   │   ├── augment.py       # Applies transforms, returns result tuples
    │   │   └── custom.py        # Custom Augmentation implementation (e.g. MixUp)
    │   └── video/
    │       └── extract.py       # OpenCV frame extraction logic
    ├── load/
    │   └── loader.py            # Saves output images and YOLO label files
    └── utils/
        └── helper.py            # Directory creation, CSV loading, pair generation
```

---

## Setup

> [!WARNING]  
> Albumentationsx requires Python >= 3.14

### 1. Clone the repository

```bash
git clone <repo-url>
cd PV-DP-HPPK
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Edit the config file

Open `src/conf/sys_config.yml` and fill in the paths and options for your environment before running anything.

---

## Configuration

All settings are in `src/conf/sys_config.yml`:

```yaml
# --- Mode ---
mode:           ""    # "strict" or "normal"
                      # strict mode will raise error if output folder already exists
                      # normal mode will not raise error

# --- Frame Extraction ---
video_dir:      ""    # Root folder containing subfolders of .mp4 videos (one subfolder per part)
frame_save_dir: ""    # Where extracted frames will be saved
csv_path:       ""    # Path to a CSV file mapping folder names to HP part names
frame_rate:     0   # Seconds between extracted frames (e.g. 0.25 = 4 fps, 0.8 = ~1.25 fps)

# --- Augmentation ---
yolo_dir:       ""    # Input: annotated YOLO dataset directory (must contain images/ and labels/)
yolo_save_dir:  ""    # Output: where augmented dataset will be saved
task:           ""    # "Detection" or "Classification"
aug_mult:       0     # Number of time an albumentations pipeline is applied
copy_mult:      0     # Number of time images and labels are copied 
mixup_mult:     0     # Number of time MixUp transform is applied 
mosaic_mult:    0     # Number of time Mosaic transform is applied 
```

### Expected input structure (for video frame extraction)

```
file_dir/
├── Part01/
│   ├── vid001.mp4
│   └── vid002.mp4
├── Part02/
│   ├── vid001.mp4
│   ├── vid002.mp4   
│   └── vid003.mp4
└── meta.csv

```

### CSV format (for frame extraction)

The CSV file maps raw video subfolder names to readable HP part names.  
The first column should be the folder name, the second column the part name:

```
P001,SVC_HP LaserJet Fuser 220V Kit
P002,SVC_HP LaserJet CYM Managed Imaging Drum
...
```

### Expected input structure (for classification augmentation)

```
file_dir/
├── Part01/
│   ├── img001.jpg
│   └── img002.jpg
├── Part02/
│   ├── img001.jpg
│   ├── img002.jpg   
│   └── img003.jpg
└── Part03/
    ├── img001.jpg   
    └── img002.jpg
```

### Expected input structure (for object detection augmentation)

```
file_dir/
├── images/
│   ├── img001.jpg
│   └── img002.jpg
└── labels/
    ├── img001.txt    # YOLO format: class cx cy w h (one object per line)
    └── img002.txt
```

---

## Usage

All commands are run from inside the `src/` directory:

```bash
cd src
```

### Stage 1 — Extract frames from videos

```bash
python main.py --option extract
```

This scans `video_dir` for subfolders, finds `.mp4` files inside each, extracts one frame every `frame_rate` seconds, and saves them to `frame_save_dir/<part_folder>/`.  
Filenames are generated automatically using UUID to avoid collisions.

### Stage 2 — Annotate extracted frames

Use an external annotation tool such as **Roboflow** or **AnyLabeling** to label the extracted frames in YOLO format before running augmentation.

### Stage 3 — Augment the annotated dataset

For a **Detection** task (bounding boxes preserved):

```bash
python main.py --option augment
# Requires: task: "Detection" in sys_config.yml
```

For a **Classification** task (no bounding boxes):

```bash
python main.py --option augment
# Requires: task: "Classification" in sys_config.yml
```

---

## Pipeline Details

### Frame Extraction

1. Loads the CSV to build a `folder_name → part_name` mapping.
2. Walks `video_dir`, finds each `.mp4` file in each subfolder.
3. Opens each video with OpenCV; extracts a frame every `frame_rate` seconds.
4. Saves each frame as a `.jpg` with the part name embedded in the filename.
5. All videos are processed in **parallel** using Python `multiprocessing`.

### Detection Augmentation

1. Reads all images and their YOLO label files from `yolo_dir`.
2. Randomly pairs each image with another image (for MixUp).
3. For each image:
   - Optionally copies the original unchanged (`copy: True`).
   - Generates `multiplier` augmented variants using: horizontal flip, brightness/contrast, rotation (±60°), Gaussian blur, Gaussian noise, hue/saturation shift.
   - Optionally applies **MixUp**: blends two images together with a random alpha (0.6–0.7) and merges their bounding box lists.
4. Saves output images and updated `.txt` label files with UUID filenames.
5. Bounding boxes are validated — any result where all boxes are lost (e.g. cropped out) is discarded.
6. Runs in **parallel** using Python `multiprocessing`.

### Classification Augmentation

Same as Detection but without bounding box handling:
1. Reads images from subdirectory-per-class folder structure.
2. Optionally copies the original.
3. Generates `multiplier` augmented variants using: horizontal flip, brightness/contrast, rotation (±30°), Gaussian blur, Gaussian noise.
4. Saves output images with UUID filenames into matching class subdirectories.
5. Runs in **parallel** using Python `multiprocessing`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Video reading and frame extraction |
| `Pillow` | Image loading and saving |
| `numpy` | Array operations for MixUp augmentation |
| `pandas` | Reading the HP parts name CSV file |
| `pyyaml` | Parsing `sys_config.yml` |
| `albumentationsx` | Image augmentation pipeline (flip, rotate, blur, noise, etc.) |
| `tqdm` | Progress bar for pipeline processing |

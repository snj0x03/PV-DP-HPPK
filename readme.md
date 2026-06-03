# PVision Data Pipeline

A config-driven data preparation pipeline for HP printer part image datasets.  
The primary task is **Detection** — locating multiple parts within a single frame.  
Handles four stages: **frame extraction** from raw videos, **image augmentation** on annotated datasets, **dataset splitting** into train/val/test sets, and **learning curve** subset generation.

> **Classification mode** is included for baseline comparison experiments only.  
> If your scenes contain a single isolated part per image, Classification can serve as a simpler alternative — but the main pipeline is Detection.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Quick Start — Full Workflow](#quick-start--full-workflow)
4. [Configuration](#configuration-srcconfsys_configyml)
5. [Pipeline Options](#pipeline-options)
6. [Dataset Folder Management](#dataset-folder-management)
   - [Safety warning](#safety-warning)
   - [Recommended folder naming](#recommended-folder-naming)
   - [Adding more data later](#adding-more-data-later)
7. [Pipeline Logic](#pipeline-logic)
8. [Jupyter Notebook Workflow (on AI server)](#jupyter-notebook-workflow-on-ai-server)
9. [Dependencies](#dependencies)

---

## Project Structure

```
PV-DP-HPPK/
├── requirements.txt
└── src/
    ├── main.py                        # Entry point — parses args, loads config, routes to pipeline
    ├── conf/
    │   └── sys_config.yml             # All runtime configuration lives here
    ├── reader/
    │   ├── video.py                   # Scans video folders, maps folder names → HP part names via CSV
    │   ├── image.py                   # Scans image folders for classification (one subfolder per class)
    │   └── detection.py               # Reads YOLO-format datasets (images/ + labels/)
    ├── pipeline/
    │   ├── extraction.py              # Video → frame extraction (parallel, with dedup)
    │   ├── detection.py               # YOLO detection augmentation
    │   ├── classification.py          # Classification augmentation
    │   ├── split.py                   # Splits dataset into train/val/test
    │   └── learning_curve.py          # Creates staged subsets for learning curve experiments
    ├── augment/
    │   ├── image/
    │   │   ├── presets.py             # Albumentations transform presets (augment, classify, no-op)
    │   │   ├── apply.py               # Loads images, runs transforms, returns result tuples
    │   │   ├── mixup.py               # MixUp: blend 2 images with alpha mixing
    │   │   └── mosaic.py              # Mosaic: combine 4 images into a 2×2 grid
    │   └── video/
    │       └── extract.py             # OpenCV frame extraction with optional deduplication
    ├── save/
    │   └── writer.py                  # Writes output images and YOLO label files with UUID filenames
    └── utils/
        ├── helpers.py                 # Shared helpers: dir creation, CSV loading, pair/group generation
        ├── dedup.py                   # Average-hash near-duplicate frame filtering
        ├── stats.py                   # Class distribution reports and imbalance warnings
        └── split.py                   # random_split and chunk_split logic
```

---

## Setup

```bash
git clone <repo-url>
cd PV-DP-HPPK
pip install -r requirements.txt
cd src
```

Edit `src/conf/sys_config.yml` with your local paths before running anything (see [Configuration](#configuration-srcconfsys_configyml)).

> **Config validation:** The pipeline checks all required paths and values before starting. If something is missing or invalid, it prints a clear error and exits — no partial runs.
>
> ```
> [CONFIG ERROR] Cannot run 'split' — fix the following in sys_config.yml:
>   'yolo_dir' is empty — set it in sys_config.yml
>   'split_ratios' must sum to 1.0, got 1.100
> ```

---

## Quick Start — Full Workflow

> Follow these steps in order. Each step produces output that the next step uses as input.

### Step 1 — Fill in `sys_config.yml`

Open `src/conf/sys_config.yml` and set all empty path fields:

```yaml
video_dir:      "C:/datasets/raw_videos"      # folder containing subfolders of .mp4 files
frame_save_dir: "C:/datasets/frames_v1"       # where extracted frames will be saved

yolo_dir:       "C:/datasets/annotated"       # annotated dataset (input for augment/split/lc)
yolo_save_dir:  "C:/datasets/augmented_v1"    # where pipeline output will be saved

task: "Detection"  # Detection is the primary task
```

### Step 2 — Extract frames from videos

```bash
python main.py -o extract
```

Frames are saved to `frame_save_dir/`. Near-duplicate frames are automatically filtered.

### Step 3 — Annotate the extracted frames

Use an external annotation tool such as **Roboflow** or **AnyLabeling** to label the frames in YOLO format. Point `yolo_dir` to the annotated dataset before the next step.

### Step 4 — Augment the annotated dataset

```bash
python main.py -o augment
```

`task: "Detection"` is the default. Enable `mosaic: True` to generate synthetic multi-part scenes from single-part annotated images. Augmented images are saved to `yolo_save_dir/`.

### Step 5 — Split into train / val / test

```bash
python main.py -o split
```

Point `yolo_dir` to the augmented output from Step 4. Results are saved to `yolo_save_dir/train/`, `val/`, `test/`.

### Step 6 (optional) — Create learning curve datasets

```bash
python main.py -o lc
```

Creates staged subsets (`stage_50/`, `stage_100/`, etc.) for incremental training experiments.

---

## Configuration (`src/conf/sys_config.yml`)

```yaml
# ── EXTRACTION ─────────────────────────────────────────
video_dir:      ""    # root folder containing subfolders of .mp4 videos (one subfolder per part)
frame_save_dir: ""    # where extracted frames will be saved
csv_path:       ""    # required only when video_type: "single" (col0: folder name, col1: part name)
frame_rate:     0.8   # seconds between extracted frames (0.8 ≈ 1.25 fps)
video_type: "single"  # "single" — one part per frame → CSV mapping, {part_name}-{uuid}.jpg
                      # "multi"  — multiple parts per frame → no CSV, {uuid}.jpg

# ── AUGMENTATION ───────────────────────────────────────
yolo_dir:       ""    # input: annotated dataset directory
yolo_save_dir:  ""    # output: where augmented / split / lc results will be saved
task:   "Classification"  # "Detection" or "Classification"
copy:   False         # if True, copy each original image to output unchanged
multiplier: 3         # augmented variants per image
mixup:  False         # MixUp: blend 2 images with random alpha (Detection only)
mosaic: False         # Mosaic: combine 4 images into a 2×2 grid (Detection only)

# ── DEDUPLICATION ──────────────────────────────────────
deduplicate:    True  # remove near-duplicate frames during extraction
hash_size:      8     # average-hash grid size (8 = 64-bit hash)
hash_threshold: 5     # Hamming distance threshold (0 = exact match only, 5 = near-duplicates)

# ── SPLIT ──────────────────────────────────────────────
split_mode:   "chunk"           # "chunk" (recommended for video data) or "random"
split_ratios: [0.7, 0.15, 0.15] # train / val / test
split_seed:   42
chunk_size:   10      # consecutive frames per chunk (chunk mode only)

# ── CLASS IMBALANCE ────────────────────────────────────
imbalance_ratio_warn: 3.0  # warn if max:min class count ratio exceeds this

# ── LEARNING CURVE ─────────────────────────────────────
lc_stages: [50, 100, 150, 300]  # images per class per stage
lc_seed:   42
lc_source_dir: ""  # optional: path to augmented output to run lc on augmented data
                   # leave empty → falls back to yolo_dir automatically
```

### CSV format

The CSV file has no header. Column 0 is the video subfolder name, column 1 is the HP part name used when saving frames:

```
P001,SVC_HP LaserJet Fuser 220V Kit
P002,SVC_HP LaserJet CYM Managed Imaging Drum
```

### Video upload rules

Extraction behavior differs by `video_type` setting. This is independent of `task` — you can extract in either mode and train a Detection model either way.

#### `video_type: "single"` — 프레임당 부품 1개 (default)

CSV로 폴더명 → 부품명 매핑. 파일명에 부품명이 포함되어 어노테이션 시 어떤 부품인지 바로 식별 가능.

```
video_dir/
├── P001/            ← folder name must match CSV col0
│   ├── clip1.mp4
│   └── clip2.mp4
└── P002/
    └── recording.mp4
```

After extraction, frames are saved as `{part_name}-{uuid}.jpg`:

```
frame_save_dir/
├── P001/
│   └── SVC_HP LaserJet Fuser 220V Kit-3f2a1c.jpg
└── P002/
    └── SVC_HP LaserJet CYM Managed Imaging Drum-7a1f3c.jpg
```

| Item | Rule |
|------|------|
| **Folder name** | Must exactly match CSV `col0` |
| **Video filename** | Free — any `.mp4` filename works |
| **Folder not in CSV** | Silently skipped (no error) |

#### `video_type: "multi"` — 프레임당 부품 2개 이상

CSV 불필요. 파일명은 UUID만 — 클래스 정보는 bbox 어노테이션 단계에서 label 파일에 기록.

```
video_dir/
├── scene_01/        ← folder name used as output subfolder only
│   └── clip.mp4
└── scene_02/
    └── clip.mp4
```

After extraction, frames are saved as `{uuid}.jpg`:

```
frame_save_dir/
├── scene_01/
│   ├── 3f2a1c.jpg
│   └── 9d4b2e.jpg
└── scene_02/
    └── 7a1f3c.jpg
```

After extraction, annotate the frames with bounding boxes using Roboflow or AnyLabeling, then point `yolo_dir` to the exported YOLO-format dataset before running `augment`.

---

## Pipeline Options

All commands are run from `src/`:

```bash
cd src
```

| Option | Command | Description |
|--------|---------|-------------|
| `extract` | `python main.py -o extract` | Extract frames from `.mp4` videos, filter near-duplicates |
| `augment` | `python main.py -o augment` | Augment annotated images (Classification or Detection) |
| `split`   | `python main.py -o split`   | Split dataset into train / val / test |
| `lc`      | `python main.py -o lc`      | Create staged subsets for learning curve experiments |

### extract

Scans `video_dir` for subfolders of `.mp4` files, extracts one frame every `frame_rate` seconds, filters near-duplicate frames (average-hash with Hamming distance), and saves to `frame_save_dir/`.  
Progress shows both video count and running frame total:

```
Videos: 3/10 [frames=847]
Frame Extraction Completed — 2341 frame(s) saved
```

- `video_type: "single"`: CSV 매핑 → `{part_name}-{uuid}.jpg`, 완료 후 클래스 분포 리포트 출력
- `video_type: "multi"`: CSV 불필요 → `{uuid}.jpg`, 클래스 리포트 없음 (bbox 어노테이션 후 확인)

### augment

**Detection** (`task: "Detection"`, default): Applies flip, brightness/contrast, rotation (±60°), blur, noise, hue/saturation shift. Bounding boxes are preserved in YOLO format; results where all boxes are lost are discarded.

- **MixUp** (`mixup: True`): Blends 2 images with a random alpha (0.6–0.7) and merges their bbox lists.
- **Mosaic** (`mosaic: True`): Combines 4 random images into a 2×2 grid — effectively simulates multi-part scenes from single-part annotated data. Bounding boxes are remapped and clipped to the canvas.

**Classification** (`task: "Classification"`, baseline only): Applies flip, brightness/contrast, rotation (±30°), blur, noise. No bbox handling.

After augmentation, prints a class distribution report. For Detection, the report shows **bbox counts per class_id** (read from label files) instead of image counts per folder, and a drop summary is shown if any augmentations were discarded:

```
Augmentation Completed — 1183/1200 saved (17 dropped, 1.4%: bboxes lost or transform error)
```

If any class has 0 images, a dedicated warning is shown instead of the ratio:

```
  part_B                                  : 0  ← EMPTY
  WARNING: 1 class(es) with 0 images: part_B
  (Max:Min ratio skipped — remove or populate empty classes before training)
```

### split

Behavior depends on `task`:

**Detection** (`task: "Detection"`, default): reads `yolo_dir/images/` + `yolo_dir/labels/`, splits image+label pairs together into `train/images/`, `train/labels/`, `val/…`, `test/…` under `yolo_save_dir`.

**Classification** (`task: "Classification"`, baseline only): reads `yolo_dir/<class>/` subfolders, splits into `train/<class>/`, `val/<class>/`, `test/<class>/` under `yolo_save_dir`. Prints class distribution and split distribution tables; saves `class_distribution.csv` and `split_distribution.csv`.

**Split modes (both tasks):**
- **`chunk`** (recommended): Groups consecutive frames into chunks of `chunk_size`, shuffles chunks, then assigns whole chunks to splits. Prevents near-duplicate adjacent frames from leaking across train/val/test boundaries.
- **`random`**: Shuffles individual files — simpler but may leak near-duplicates.

### lc (learning curve)

Creates staged subsets under `yolo_save_dir/stage_50/`, `stage_100/`, `stage_150/`, `stage_300/`.

**Source directory:** by default reads from `yolo_dir`. To run lc on augmented data without changing `yolo_dir`, set `lc_source_dir` to the augmented output path:

```yaml
yolo_dir:       "C:/datasets/annotated"    # used by augment / split
lc_source_dir:  "C:/datasets/augmented_v1" # lc reads from here instead
yolo_save_dir:  "C:/datasets/lc_v1"        # staged subsets written here
```

**Detection** (`task: "Detection"`, default): samples `min(N, available)` image+label **pairs** per stage (N = total images). Output: `stage_N/images/` + `stage_N/labels/`.

**Classification** (`task: "Classification"`, baseline only): samples `min(N, available)` images **per class** per stage. Output: `stage_N/<class>/`.

Example with 3 classes:

| Stage folder | Images per class | Total images |
|---|---|---|
| `stage_50/` | 50 | ~150 |
| `stage_100/` | 100 | ~300 |
| `stage_150/` | 150 | ~450 |
| `stage_300/` | 300 | ~900 |

Train the model on each stage and measure validation accuracy to find the point of diminishing returns.

---

## Dataset Folder Management

**Output files are never overwritten** — each file is saved with a unique UUID filename (e.g. `part_A-3f2a1c.jpg`). Running a pipeline twice on the same `save_dir` will **add** new files on top of existing ones.

### Safety warning

If the target save directory already contains files, the pipeline will warn you before starting:

```
[WARNING] save_dir already contains 547 file(s):
  → C:/datasets/augmented_v1
  New files from 'augment' will be ADDED on top of existing files.
  To start fresh, use a different save_dir path in sys_config.yml.
Proceed? [y/N]:
```

- Enter `y` to continue and add to existing files.
- Press Enter or type `N` to abort (existing files are untouched).

### Recommended folder naming

To keep dataset versions separate, change `yolo_save_dir` (or `frame_save_dir`) each time you change settings:

```yaml
# First run — multiplier: 3
yolo_save_dir: "C:/datasets/augmented_v1"

# Second run — multiplier: 5, mosaic: True
yolo_save_dir: "C:/datasets/augmented_v2"
```

This way you can always go back to a previous dataset without re-running the pipeline.

### Adding more data later

When new videos are available, follow the appropriate case below.

**Case 1 — More videos for existing classes:**

```
1. Add new .mp4 files to the existing subfolders in video_dir
2. Change frame_save_dir to a new path (e.g. frames_v2)
   → python main.py -o extract

3. Annotate only the new frames

4. Set yolo_dir to the new annotated folder
   Set yolo_save_dir to a new path (e.g. augmented_v2)
   → python main.py -o augment

5. Set yolo_dir = augmented_v2, yolo_save_dir = split_v2
   → python main.py -o split

6. Retrain the model on split_v2
```

**Case 2 — Adding a new class (new part type):**

```
1. Create a new subfolder in video_dir matching the new CSV entry
2. Add the new part to part_names.csv
3. Extract + annotate the new class frames
4. Merge the new annotated class folder into the existing annotated dataset
5. Re-run augment → split on the full merged dataset (new save_dir path)
6. Retrain the model
```

**Key principle:** always bump the `save_dir` version when re-running so old datasets are preserved for comparison.

```
frames_v1/   augmented_v1/   split_v1/   ← first training run
frames_v2/   augmented_v2/   split_v2/   ← after adding more data
```

---

## Pipeline Logic

### Frame Extraction

```
video_dir/{part_folder}/*.mp4
    ↓  video_dataset()  — maps folder name → part name via CSV
    ↓  extract_frame()  — sample every N-th frame (interval = fps × frame_rate)
    ↓  filter_duplicates()  — skip frames within Hamming distance threshold
    ↓  save_image_random_part()  — saves as {part_name}-{uuid}.jpg
frame_save_dir/{part_folder}/
    ↓  imbalance_report()  — prints class counts + warnings
```

### Detection Augmentation

```
yolo_dir/images/ + labels/
    ↓  yolo_dataset()  — pairs each image with its YOLO label
    ↓  make_pair_list()  — randomly pairs images for MixUp
    ↓  for each pair:
         copy_transform (if copy=True) → save
         augment_transform × multiplier → save
         mixup_transform (if mixup=True) → save
         mosaic_transform (if mosaic=True) → save
    ↓  invalid results (all bboxes lost) are discarded silently
    ↓  drop summary printed: "1183/1200 saved (17 dropped, 1.4%)"
yolo_save_dir/images/ + labels/
    ↓  count_classes_detection()  — counts bbox occurrences per class_id from label files
    ↓  imbalance_report()  — warns if any class has 0 images or ratio exceeds threshold
```

### Classification Augmentation

```
yolo_dir/{class_A}/ {class_B}/ ...
    ↓  image_dataset()  — lists images per class
    ↓  for each image:
         empty_transform (if copy=True) → save
         classify_transform × multiplier → save
yolo_save_dir/{class_A}/ {class_B}/ ...
```

### Split (Classification)

```
source/{class}/[frame_0001.jpg ... frame_0300.jpg]  ← sorted by filename
    ↓  group into chunks of chunk_size (e.g. 10)
    ↓  shuffle chunks (seed-controlled)
    ↓  assign chunks to train/val/test at ratio boundaries
save_dir/train/{class}/  val/{class}/  test/{class}/
    ↓  split_distribution_report()  — prints per-class counts in each split
```

### Split (Detection)

```
source/images/ + labels/
    ↓  collect all image paths (sorted)
    ↓  chunk-shuffle or random-shuffle
    ↓  assign to train/val/test at ratio boundaries
    ↓  copy image + matching label file together
save_dir/train/images/  train/labels/
         val/images/    val/labels/
         test/images/   test/labels/
```

Why chunk split? Consecutive video frames are nearly identical. If individual frames are shuffled randomly, the same scene can appear in both train and val — inflating validation accuracy. Chunk split keeps an entire group of similar frames in one split only.

---

## Jupyter Notebook Workflow (on AI server)

After uploading the pipeline output to the school AI server, run the following cells in a Jupyter notebook.

### Training (per stage or split)

```python
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")  # classification
# model = YOLO("yolo11n.pt")    # detection

model.train(
    data="stage_150",   # path to the stage or split dataset
    epochs=50,
    imgsz=224,
    project="runs",
    name="stage_150",
)
```

### Recording learning curve results

Run this cell after each training stage (update `stage` and the run name each time):

```python
import pandas as pd
import os

results = pd.read_csv("runs/stage_150/results.csv")
results.columns = results.columns.str.strip()

best_acc = results["metrics/accuracy_top1"].max()   # classification
# best_acc = results["metrics/mAP50"].max()         # detection

log_path = "lc_results.csv"
row = pd.DataFrame([{"stage": 150, "best_val_acc": round(best_acc, 4)}])
row.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
print(f"stage 150 → best acc: {best_acc:.4f}")
```

### Plotting the learning curve

Run once after all stages are complete:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("lc_results.csv").sort_values("stage")
print(df)

plt.figure(figsize=(7, 4))
plt.plot(df["stage"], df["best_val_acc"], marker="o")
plt.xlabel("Images per class")
plt.ylabel("Val Accuracy")
plt.title("Learning Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.show()
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Video reading and frame extraction |
| `Pillow` | Image loading and saving |
| `numpy` | Array operations (MixUp, average hash) |
| `pandas` | Reading the HP parts CSV, saving stats reports |
| `pyyaml` | Parsing `sys_config.yml` |
| `albumentations` | Augmentation pipeline (flip, rotate, blur, noise, etc.) |
| `tqdm` | Progress bars |

```bash
pip install -r requirements.txt
```

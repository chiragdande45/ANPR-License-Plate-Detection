# ANPR Phase 1 — Automatic Number Plate Detection & Enhancement

**Reference:** Batra et al., "A Novel Memory and Time-Efficient ALPR System Based on YOLOv5,"
*Sensors* 2022, 22, 5283.

> **Phase 1 only.** This codebase covers plate detection and image enhancement.
> OCR (Phase 2) is not implemented here.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1 Pipeline](#2-phase-1-pipeline)
3. [Method Decisions](#3-method-decisions)
4. [Project Structure](#4-project-structure)
5. [Installation](#5-installation)
6. [Dataset Preparation](#6-dataset-preparation)
7. [Training YOLOv11n](#7-training-yolov11n)
8. [Validation / Evaluation](#8-validation--evaluation)
9. [Phase 1 Inference](#9-phase-1-inference)
10. [Output Structure](#10-output-structure)
11. [Experiments & Method Comparison](#11-experiments--method-comparison)
12. [Configuration Reference](#12-configuration-reference)
13. [Failure Handling](#13-failure-handling)
14. [Train / Val / Test Rules](#14-train--val--test-rules)

---

## 1. Project Overview

This project implements the complete Phase 1 ANPR pipeline:

```
Full vehicle image
  → 640 px letterbox resize
  → YOLOv11n plate detection
  → Adaptive ROI padding & crop
  → Conditional perspective correction
  → Adaptive Lanczos upscaling
  → Conditional bilateral denoising
  → CLAHE (local contrast)
  → Selective glare suppression
  → Selective mud/dirt suppression
  → Stroke-width adaptive unsharp masking
  → Sauvola adaptive thresholding
  → Adaptive morphological closing
  → Final enhanced plate image
```

The same code handles any number of input images without manual tuning per image.

---

## 2. Phase 1 Pipeline

| # | Stage | Method |
|---|-------|--------|
| 1 | Letterbox resize | 640 px, aspect-ratio preserving with padding |
| 2 | Plate detection | YOLOv11n (ultralytics) |
| 3 | ROI crop | Bounding-box + adaptive 5–10% padding |
| 4 | Perspective correction | Conditional 4-point warp / rotation deskew |
| 5 | Upscaling | Adaptive Lanczos (scale derived from plate width) |
| 6 | Denoising | Conditional bilateral filter |
| 7 | Contrast | CLAHE on LAB L-channel |
| 8 | Glare | Selective V-channel highlight suppression |
| 9 | Dirt | Black-hat + connected-component shape filtering |
| 10 | Character enhancement | Stroke-width adaptive unsharp masking |
| 11 | Thresholding | Sauvola adaptive threshold |
| 12 | Morphology | Adaptive morphological closing |

---

## 3. Method Decisions

| Stage | Paper Method | Final Method | Status |
|-------|-------------|--------------|--------|
| Image Resizing | Resize to 640 px | 640 px letterbox | KEEP |
| Plate Detection | YOLOv5s | YOLOv11n | REPLACE |
| ROI Crop | Bounding-box crop | + adaptive 5–10% padding | REPLACE |
| Perspective | Not specified | Conditional 4-point warp | BASIC ADDITION |
| Upscaling | Not specified | Adaptive Lanczos | ADDITION |
| Noise Removal | Not specified | Conditional bilateral filter | ADDITION |
| Illumination | Not specified | CLAHE (LAB L-channel) | ADDITION |
| Glare | Not specified | Selective highlight suppression | ADDITION |
| Dirt | Not specified | Black-hat + shape filtering | ADDITION |
| Character Enhancement | Not specified | Stroke-width adaptive unsharp | ADDITION |
| Thresholding | Not specified | Sauvola adaptive threshold | ADDITION |
| Morphology | Not specified | Adaptive morphological closing | ADDITION |

To print the full table with reasons:

```
python main.py --method-table
```

---

## 4. Project Structure

```
ANPR/
│
├── core.py                  ← All image-processing algorithms
├── experiments.py           ← Stage comparisons, scoring, method table
├── main.py                  ← Batch pipeline, config, reporting
│
├── requirements.txt
├── dataset.yaml             ← YOLOv11n training config
├── .gitignore
│
├── dataset/
│   ├── train/
│   │   ├── images/          ← Training vehicle images
│   │   └── labels/          ← YOLO labels (.txt)
│   ├── val/
│   │   ├── images/          ← Validation images
│   │   └── labels/
│   └── test/
│       ├── images/          ← Unseen test images
│       └── labels/          ← Ground-truth labels (if available)
│
├── models/
│   └── plate_detector/
│       └── best.pt          ← Trained YOLOv11n weights (place here)
│
├── test_images/             ← Vehicle images for Phase 1 inference
│
└── outputs/
    ├── batch_report/        ← batch_report.json
    ├── car001/
    │   ├── 01_input/
    │   ├── 02_detection/
    │   ├── 03_roi/
    │   ├── 04_perspective/
    │   ├── 05_upscaling/
    │   ├── 06_denoising/
    │   ├── 07_clahe/
    │   ├── 08_glare/
    │   ├── 09_dirt/
    │   ├── 10_character/
    │   ├── 11_threshold/
    │   ├── 12_morphology/
    │   ├── final/
    │   └── report/
    └── carN/
        └── ...
```

---

## 5. Installation

### 5.1 Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 5.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 5.3 GPU support (optional but recommended for training)

Replace the CPU PyTorch lines in `requirements.txt` with the appropriate CUDA wheel,
or install separately after the base install:

**CUDA 11.8:**
```bash
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 5.4 Verify installation

```bash
python -c "import cv2, numpy, ultralytics, skimage; print('OK')"
```

---

## 6. Dataset Preparation

### 6.1 Label format

Each image in `dataset/train/images/` must have a matching `.txt` file in
`dataset/train/labels/` with the same base name.

Each label file contains one line per plate:

```
<class_id> <cx> <cy> <width> <height>
```

- All values are normalised to [0, 1] relative to image dimensions.
- `class_id` = `0` (licence_plate).

Example label for a plate occupying the centre-bottom of a 1280×720 image:

```
0 0.512 0.823 0.210 0.078
```

### 6.2 Recommended public datasets for number-plate detection

- **OpenALPR benchmark dataset**
- **CCPD (Chinese City Parking Dataset)**
- **UFPR-ALPR dataset**
- **Kaggle: Car License Plate Detection**

Download images, add labels in YOLO format, then split into `train/val/test`.

### 6.3 Splitting guidance

A typical split for a small dataset:

| Split | Fraction | Purpose |
|-------|----------|---------|
| train | 70% | Model learning |
| val | 15% | Hyperparameter selection, early stopping |
| test | 15% | Final unbiased evaluation only |

**Never copy test images into train or val.**

---

## 7. Training YOLOv11n

### 7.1 Verify dataset.yaml

Open `dataset.yaml` and confirm the `path` field points to the ANPR project root:

```yaml
path: d:/ANPR
```

### 7.2 Training command

```bash
yolo detect train \
    model=yolo11n.pt \
    data=dataset.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    name=plate_detector \
    project=runs/detect \
    patience=20 \
    optimizer=AdamW \
    lr0=0.001 \
    weight_decay=0.0005 \
    augment=True \
    hsv_h=0.015 \
    hsv_s=0.7 \
    hsv_v=0.4 \
    flipud=0.0 \
    fliplr=0.5 \
    mosaic=1.0 \
    close_mosaic=10
```

**Windows PowerShell (single line):**
```powershell
yolo detect train model=yolo11n.pt data=dataset.yaml epochs=100 imgsz=640 batch=16 name=plate_detector project=runs/detect patience=20 optimizer=AdamW lr0=0.001
```

### 7.3 Key parameter notes

| Parameter | Value | Reason |
|-----------|-------|--------|
| `model` | `yolo11n.pt` | Downloads YOLOv11n pretrained weights automatically |
| `imgsz` | `640` | Matches the letterbox size used in inference |
| `batch` | `16` | Reduce to `8` if GPU memory is limited |
| `patience` | `20` | Early stopping after 20 epochs without val improvement |
| `augment` | `True` | Enables built-in augmentations for better generalisation |
| `flipud` | `0.0` | Plates are never upside-down — disable vertical flip |

### 7.4 After training

Ultralytics saves weights to:
```
runs/detect/plate_detector/weights/best.pt
```

Copy the best weights to the models folder:

```bash
# Windows
copy runs\detect\plate_detector\weights\best.pt models\plate_detector\best.pt

# Linux / macOS
cp runs/detect/plate_detector/weights/best.pt models/plate_detector/best.pt
```

The inference pipeline will automatically load `models/plate_detector/best.pt`.

---

## 8. Validation / Evaluation

### 8.1 Validate on the validation split (during development)

```bash
yolo detect val \
    model=runs/detect/plate_detector/weights/best.pt \
    data=dataset.yaml \
    split=val \
    imgsz=640
```

### 8.2 Evaluate on the test split (final unbiased evaluation)

```bash
yolo detect val \
    model=models/plate_detector/best.pt \
    data=dataset.yaml \
    split=test \
    imgsz=640 \
    name=test_eval \
    project=runs/detect
```

This reports mAP@0.5, mAP@0.5:0.95, Precision, and Recall for the test split.

**Only run this once** on the final model. Do not use test results to retune the model.

### 8.3 Evaluate Phase 1 pipeline with ground-truth labels

After placing ground-truth YOLO labels for your test images in `dataset/test/labels/`:

```bash
python main.py \
    --input dataset/test/images \
    --labels dataset/test/labels \
    --weights models/plate_detector/best.pt
```

This computes Precision, Recall and mean IoU for the pipeline's plate detections
and saves the results in `outputs/batch_report/batch_report.json`.

If ground-truth labels are unavailable, omit `--labels`. The batch report will
contain confidence values and qualitative results instead.

---

## 9. Phase 1 Inference

### 9.1 Default — process the `test_images/` folder

Place vehicle images (`.jpg`, `.jpeg`, `.png`, `.bmp`) in `test_images/`, then:

```bash
python main.py
```

### 9.2 Custom input folder

```bash
python main.py --input path/to/folder
```

### 9.3 Single image

```bash
python main.py --input test_images/car001.jpg
```

### 9.4 Custom weights

```bash
python main.py --input test_images --weights models/plate_detector/best.pt
```

### 9.5 Custom output folder

```bash
python main.py --input test_images --output my_results
```

### 9.6 Adjust detection confidence threshold

```bash
python main.py --input test_images --conf 0.35
```

### 9.7 Full options

```
python main.py --help

  --input   / -i    Image file or folder (default: test_images/)
  --weights / -w    YOLOv11n weights path (default: models/plate_detector/best.pt)
  --output  / -o    Output base folder (default: outputs/)
  --conf            YOLO confidence threshold (default: 0.25)
  --labels          Folder of YOLO ground-truth labels for evaluation
  --method-table    Print method replacement table and exit
```

---

## 10. Output Structure

For each input image `carXXX.jpg`, the pipeline creates:

```
outputs/
└── carXXX/
    ├── 01_input/
    │   └── original.jpg
    ├── 02_detection/
    │   └── detected_plate.jpg        ← bounding box + conf + dimensions
    ├── 03_roi/
    │   └── plate_roi.jpg             ← cropped plate with adaptive padding
    ├── 04_perspective/
    │   ├── plate_before_correction.jpg
    │   └── plate_corrected.jpg
    ├── 05_upscaling/
    │   └── plate_upscaled.jpg
    ├── 06_denoising/
    │   └── plate_denoised.jpg
    ├── 07_clahe/
    │   └── plate_clahe.jpg
    ├── 08_glare/
    │   ├── pre_glare.jpg
    │   ├── glare_mask.jpg
    │   └── plate_glare_suppressed.jpg
    ├── 09_dirt/
    │   ├── pre_dirt.jpg
    │   ├── dirt_mask.jpg
    │   └── dirt_suppressed.jpg
    ├── 10_character/
    │   └── character_enhanced.jpg
    ├── 11_threshold/
    │   └── sauvola.jpg
    ├── 12_morphology/
    │   └── morphology_closed.jpg
    ├── final/
    │   ├── final_enhanced.jpg        ← binary (Sauvola + closing)
    │   └── final_enhanced_continuous.jpg  ← continuous-tone alternative
    └── report/
        └── decision_report.json      ← per-image metrics and decisions

outputs/batch_report/
└── batch_report.json                 ← batch-level summary
```

### decision_report.json fields

```json
{
  "image": "car001.jpg",
  "plate_detected": true,
  "detection_confidence": 0.872,
  "plate_width_px": 310.4,
  "plate_height_px": 88.2,
  "aspect_ratio": 3.519,
  "roi_padding_used": 0.07,
  "perspective_corrected": true,
  "skew_angle_deg": -4.3,
  "perspective_method": "4-point perspective warp",
  "upscale_factor": 1.289,
  "denoising_applied": false,
  "noise_score": 0.003,
  "laplacian_variance": 312.8,
  "illumination_method": "CLAHE (clip=2.0, tile=[8, 8])",
  "glare_method": "Selective highlight suppression (V-channel mask)",
  "glare_present": false,
  "dirt_method": "Black-hat + connected-component shape/edge filtering",
  "dirt_found": false,
  "character_enhancement": "Stroke-width adaptive unsharp masking",
  "threshold_method": "Sauvola adaptive threshold (skimage)",
  "morphology_method": "Adaptive morphological closing",
  "processing_time_s": 0.412,
  "success": true,
  "error": null,
  "stage_timings_s": { ... }
}
```

---

## 11. Experiments & Method Comparison

Run side-by-side comparisons for any pipeline stage on a single image:

```bash
# Compare all stages at once
python experiments.py --image test_images/car001.jpg --stage all

# Compare only CLAHE clip limits
python experiments.py --image test_images/car001.jpg --stage clahe

# Compare bilateral filter sigma values
python experiments.py --image test_images/car001.jpg --stage bilateral

# Compare glare suppression thresholds
python experiments.py --image test_images/car001.jpg --stage glare

# Compare unsharp masking amount values
python experiments.py --image test_images/car001.jpg --stage unsharp

# Compare upscaling interpolation methods and scales
python experiments.py --image test_images/car001.jpg --stage upscaling

# Compare thresholding methods (Otsu / Adaptive Mean / Adaptive Gaussian / Sauvola)
python experiments.py --image test_images/car001.jpg --stage threshold

# Compare morphological closing kernel sizes
python experiments.py --image test_images/car001.jpg --stage morphology

# Print method replacement table only
python experiments.py --image dummy.jpg --stage table
```

Experiment outputs are saved to `outputs/experiments/<image_stem>/`.
Each stage produces individual candidate images and a side-by-side comparison grid.

---

## 12. Configuration Reference

All parameters are in `DEFAULT_CONFIG` at the top of `main.py`.
No source code needs to change for parameter tuning — only the config dict.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf_threshold` | `0.25` | YOLO detection minimum confidence |
| `letterbox_size` | `640` | Detector input size |
| `roi_pad_ratio` | `0.07` | Plate crop padding fraction (5–10%) |
| `skew_threshold` | `3.0` | Min skew (°) to trigger perspective correction |
| `upscale_min_width` | `400` | Target plate width after upscaling (px) |
| `upscale_max_scale` | `4.0` | Maximum upscale factor |
| `noise_threshold` | `0.01` | Inverse-variance score above which bilateral is applied |
| `bilateral_d` | `9` | Bilateral filter neighbourhood diameter |
| `bilateral_sigma_color` | `40` | Bilateral colour sigma |
| `bilateral_sigma_space` | `40` | Bilateral spatial sigma |
| `clahe_clip` | `2.0` | CLAHE clip limit |
| `clahe_tile` | `[8,8]` | CLAHE tile grid size |
| `glare_threshold` | `240` | V-channel value above which pixel is glare |
| `glare_blend` | `0.7` | Glare correction blend factor |
| `dirt_area_min` | `10` | Minimum dirt blob area (px²) |
| `dirt_area_max_ratio` | `0.04` | Maximum dirt blob area as fraction of ROI |
| `dirt_edge_thresh` | `0.15` | Edge density above which blob is protected as character |
| `unsharp_amount` | `1.2` | Unsharp mask sharpening strength |
| `unsharp_threshold` | `3` | Min pixel difference to apply unsharp correction |
| `sauvola_window` | `25` | Sauvola local window size |
| `sauvola_k` | `0.2` | Sauvola sensitivity |

---

## 13. Failure Handling

- If YOLOv11n detects no plate: the image is recorded as a failed detection, a
  failure entry is added to the batch report, and processing continues with the
  next image. No crash.
- If any processing stage raises an exception: the error is caught, recorded in
  the per-image report with the error message, and the pipeline moves on.
- All failures are summarised in `outputs/batch_report/batch_report.json`.

---

## 14. Train / Val / Test Rules

| Split | Folder | Purpose |
|-------|--------|---------|
| Train | `dataset/train/` | YOLOv11n weight learning |
| Val | `dataset/val/` | Model selection, early stopping, hyperparameter tuning |
| Test | `dataset/test/` | Final unbiased evaluation — use only once on the final model |
| Inference | `test_images/` | Phase 1 pipeline testing on unseen vehicle images |

**Rules:**
- Never train on test images.
- Never use test results to tune the model.
- Never copy test images into train or val.
- `test_images/` is for pipeline inference only — it is not a training resource.

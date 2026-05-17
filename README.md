# 🧠 Crowd Vision 
Real-Time Crowd Detection & Risk Prediction

A real-time crowd monitoring system built on **YOLOv8m** that detects people in images, video files, and live webcam feeds — and automatically predicts crowd risk levels (LOW / MEDIUM / HIGH / CRITICAL) based on people count and frame density.

---

## 📸 What It Does

- Detects and counts people in real time using a custom-trained YOLOv8m model
- Calculates crowd density as a percentage of frame area covered
- Assigns a **risk level** based on both head count and density
- Displays live overlay with bounding boxes, count, density, and risk recommendation
- Supports webcam, video files, and single images

---

## 🏗️ Model Details

| Property | Value |
|---|---|
| Architecture | YOLOv8m (medium, 25M parameters) |
| Dataset | Crowd Density Estimation — Roboflow |
| Training images | ~1184 |
| Validation images | ~167 |
| Image size | 640×640 |
| Optimizer | AdamW |
| Target mAP50 | 0.75 |
| Device | Apple M4 (MPS backend) |

---

## ⚠️ Risk Level System

Risk is determined by two signals combined:

**1. Person count**
| Count | Risk Level |
|---|---|
| 0 – 5 | 🟢 LOW |
| 6 – 15 | 🟠 MEDIUM |
| 16 – 30 | 🔴 HIGH |
| 31+ | 🚨 CRITICAL |

**2. Bounding box density** (% of frame area covered)

Density upgrades the risk level when people overlap and the model under-counts:

| Density | Minimum Risk |
|---|---|
| ≥ 10% | MEDIUM |
| ≥ 25% | HIGH |
| ≥ 50% | CRITICAL |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/crowd-vision.git
cd crowd-vision
```

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install ultralytics opencv-python torch
```

### 3. Run inference

```bash
# Live webcam
python infer.py webcam

# Video file
python infer.py crowd_video.mp4

# Single image
python infer.py crowd_photo.jpg
```

---

## 🧪 Run on Test Images

```bash
python test.py
```

Results are saved to `./runs/test_results/` with bounding boxes and risk overlays drawn on each image.

Or use the model directly in your own code:

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model.predict(source="your_image.jpg", conf=0.25)

for r in results:
    print(f"People detected: {len(r.boxes)}")
```

---

## 🏋️ Train From Scratch

If you want to retrain on your own dataset:

```bash
# 1. Prepare dataset in YOLO format:
# dataset/
#   images/train/   ← training images
#   images/val/     ← validation images
#   labels/train/   ← YOLO .txt label files
#   labels/val/

# 2. Run training
python train.py
```

Training will automatically stop when mAP50 reaches the target (default: 0.75).

Key config options at the top of `train.py`:

```python
MODEL_SIZE   = "yolov8m.pt"   # nano/small/medium/large
EPOCHS       = 100
BATCH_SIZE   = 8              # reduce to 4 if out of memory
MAP50_TARGET = 0.75           # auto-stop when this accuracy is reached
```

---

## 📁 Project Structure

```
crowd-vision/
├── train.py          # YOLOv8 training script (MPS/CUDA/CPU)
├── infer.py          # Real-time inference with risk overlay
├── test.py           # Batch test on a folder of images
├── weights/
│   └── best.pt       # Trained model weights
└── README.md
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 training & inference |
| `opencv-python` | Video/image processing & overlay |
| `torch` | Deep learning backend (MPS/CUDA/CPU) |
| `PyYAML` | Dataset config parsing |

Install all at once:
```bash
pip install ultralytics opencv-python torch pyyaml
```

---

## 🔧 Configuration

Tune the risk thresholds in `infer.py` to match your camera's field of view:

```python
# Wider FOV (drone, CCTV) → raise these numbers
COUNT_LOW      = 5
COUNT_MEDIUM   = 15
COUNT_HIGH     = 30

# Confidence threshold — lower catches more people, higher reduces false positives
CONF_THRESH = 0.35
```

---

## 📄 License

This project is for educational and research purposes.
Dataset: [Crowd Density Estimation](https://universe.roboflow.com/mohd-zeeshan-jguor/crowd-density-estimation) — CC BY 4.0

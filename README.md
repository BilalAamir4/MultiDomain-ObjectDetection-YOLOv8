# 🔍 AI-Based Multi-Environment Object Detection System

> YOLOv8s fine-tuned on COCO 2017 for real-time object detection across Traffic, Home, Wildlife, and Classroom environments — with a live Gradio web interface.

**Course:** CSC411 – Artificial Intelligence | Bahria University, Islamabad  
**Semester:** Spring 2025 – BSIT 6B  
**Instructor:** Ms. Sidra Ayesha

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Web Application](#web-application)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Team](#team)

---

## Project Overview

This project implements a multi-environment object detection system using **YOLOv8s** fine-tuned on the **COCO 2017 validation dataset**. The system detects objects across four real-world environments and automatically classifies images into the most likely environment.

**Supported Environments:**

| Environment | Example Objects |
|-------------|-----------------|
| 🐘 Wildlife | Bear, Giraffe, Zebra, Horse, Elephant |
| 🚗 Traffic | Car, Bus, Truck, Bicycle, Traffic Light |
| 🏠 Home | Laptop, TV, Chair, Cup, Bed |
| 📚 Classroom | Backpack, Book, Scissors, Keyboard, Mouse |

The system runs in real time at **9.3ms per image** (>100 FPS on GPU) and is accessible through a clean Gradio web interface — no coding required to use it.

---

## Results

### Overall Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | **52.35%** |
| mAP@0.5:0.95 | 39.69% |
| Precision | 65.33% |
| Recall | 56.96% |
| Inference Speed | 9.3 ms/image |
| Model Size | 22.6 MB |

### Per-Environment Performance

| Environment | mAP@0.5 | Precision | Recall | F1-Score |
|-------------|---------|-----------|--------|----------|
| 🐘 Wildlife | **82.75%** | 87.54% | 83.86% | 85.66% |
| 🚗 Traffic | 57.09% | 70.71% | 59.80% | 64.80% |
| 🏠 Home | 50.51% | 63.71% | 55.34% | 59.23% |
| 📚 Classroom | 37.75% | 54.49% | 41.00% | 46.79% |

### Top Performing Classes

| Class | mAP@0.5 | Environment |
|-------|---------|-------------|
| Stop Sign | 99.5% | Traffic |
| Microwave | 99.5% | Home |
| Bear | 98.2% | Wildlife |
| Giraffe | 94.5% | Wildlife |
| Horse | 90.8% | Wildlife |

---

## Project Structure

```
MultiDomain-ObjectDetection-YOLOv8/
│
├── data/
│   ├── kaggle.py                 # Downloads COCO 2017 val set via Kaggle API
│   ├── convert_annotations.py    # Converts COCO JSON → YOLO .txt format
│   ├── split_dataset.py          # Splits images 70/20/10 (train/val/test)
│   └── dataset.yaml              # YOLOv8 config: paths, 80 class names
│
├── model/
│   ├── train.py                  # Fine-tunes YOLOv8s on the prepared dataset
│   └── best.pt                   # Trained model weights (22.6 MB)
│
├── results/
│   ├── predict.py                # Runs detection on all 496 test images
│   ├── evaluate.py               # Computes mAP, precision, recall, F1 per environment
│   ├── plot_results.py           # Generates performance bar charts
│   ├── metrics.csv               # Saved evaluation metrics
│   └── plots/                    # Generated charts (3 files)
│
└── app/
    └── app.py                    # Gradio web application
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.1 (recommended; CPU works but is slower)
- Kaggle API credentials (for dataset download)

### 1. Clone the Repository

```bash
git clone https://github.com/BilalAamir4/MultiDomain-ObjectDetection-YOLOv8.git
cd MultiDomain-ObjectDetection-YOLOv8
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv yolo_env

# Windows
yolo_env\Scripts\activate

# macOS / Linux
source yolo_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install ultralytics gradio torch torchvision pandas matplotlib kaggle
```

### 4. Download the Dataset

Place your `kaggle.json` API credentials in `~/.kaggle/`, then run:

```bash
python data/kaggle.py
```

### 5. Prepare the Dataset

```bash
python data/convert_annotations.py   # Convert COCO annotations to YOLO format
python data/split_dataset.py         # Split into train / val / test
```

---

## Usage

### Training

To fine-tune the model on the prepared dataset:

```bash
python model/train.py
```

The best model weights are saved to `model/best.pt`. Training takes approximately **36 minutes** on an RTX 3050 6GB GPU.

> **Note:** Training uses early stopping with patience=10. Because we start from COCO-pretrained weights, the model typically converges by epoch 1 and stops around epoch 11. This is expected behaviour.

### Evaluation

Run detection on the test set and compute metrics:

```bash
python results/predict.py     # Run detection on 496 test images
python results/evaluate.py    # Compute per-class and per-environment metrics
python results/plot_results.py # Generate performance charts
```

Metrics are saved to `results/metrics.csv`. Charts are saved to `results/plots/`.

---

## Web Application

### Launch

```bash
python app/app.py
```

Wait for `Model ready!` to appear in the terminal, then open:

```
http://127.0.0.1:7860
```

A public shareable URL (valid for 72 hours) is also printed to the terminal via Gradio's built-in tunneling.

### Features

- **Image upload** — drag-and-drop or click to browse (JPG, PNG supported)
- **Confidence slider** — adjust detection threshold from 0.1 to 0.9
- **Annotated output** — bounding boxes with class labels and confidence scores
- **Environment classification** — automatically identifies Traffic, Home, Wildlife, or Classroom
- **Detection summary** — lists all detected objects with counts and top confidence scores
- **Mobile compatible** — accessible from any device on the same network

> **Tip:** If detections look sparse, lower the confidence threshold to 0.15 to see more results.

---

## Dataset

**Source:** [COCO 2017 Validation Set](https://cocodataset.org)

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 3,466 | 70% |
| Validation | 990 | 20% |
| Test | 496 | 10% |
| **Total** | **4,952** | — |

- 5,000 images downloaded; 4,952 retained (images with at least one annotation)
- 80 object categories
- Annotations converted from COCO JSON format to YOLO normalized `.txt` format

---

## Model Architecture

**Model:** YOLOv8s (Small variant by Ultralytics)

| Component | Description |
|-----------|-------------|
| Backbone | CSPDarknet — extracts hierarchical visual features |
| Neck | Feature Pyramid Network (FPN) — multi-scale feature fusion |
| Head | Anchor-free detection head |
| Parameters | 11,166,560 |
| GFLOPs | 28.8 |
| Input Size | 640 × 640 px |

**Training configuration:**

| Hyperparameter | Value |
|----------------|-------|
| Base weights | `yolov8s.pt` (COCO pretrained) |
| Epochs | 50 (early stopped at 11) |
| Batch size | 16 |
| Optimizer | AdamW (auto-tuned) |
| Learning rate | 0.000119 |
| Augmentation | Mosaic, flipping, scaling, color jitter |
| GPU | NVIDIA RTX 3050 6GB |
| Training time | ~36 minutes |

---

## Known Limitations

- Static image detection only — live video is not supported
- Some animals not in COCO (lion, tiger) may be misclassified as similar categories
- Small objects (scissors, books, spoons) are detected less reliably
- Performance degrades on heavily filtered or stylized images

---

## References

- Redmon et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR.
- Lin et al. (2014). *Microsoft COCO: Common Objects in Context.* ECCV.
- Ultralytics. (2023). [YOLOv8 Documentation](https://docs.ultralytics.com)
- [COCO Dataset](https://cocodataset.org)
- [Gradio](https://gradio.app)

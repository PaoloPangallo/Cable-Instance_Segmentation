# ⚡ Electric Cable Segmentation — *Computer Vision Project*

<p align="center">
  <img src="https://img.shields.io/badge/Computer%20Vision-Instance%20Segmentation-00f2ff?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Thin%20Objects-Challenging-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Transformer-RF--DETR-purple?style=for-the-badge" />
</p>

<p align="center">
  <b>Instance Segmentation of electric cables in aerial imagery</b><br>
  An in-depth analysis of <i>Transformer-based</i> architectures for thin-object detection
</p>

---

## 🧠 Overview

This project tackles one of the most challenging problems in **applied Computer Vision**: **electric cable instance segmentation**, where objects are:

* extremely **thin (thin objects)**
* **elongated** with very small pixel area
* often **occluded** or visually confused with background structures

The goal is to evaluate the effectiveness of **RF-DETR** and **Transformer-based models** compared to traditional CNN-based approaches.

---

## 👨‍🎓 Authors

| Name                   | Student ID |
| ---------------------- | ---------- |
| **Paolo Pangallo**     | 263594     |
| **Gianluigi Oricchio** | 269674     |

📍 *University of Calabria*
📆 *Academic Year 2025 / 2026*

---

## 🚧 Problem Statement — Why is it hard?

> Electric cables are not standard vision objects.

### ❌ Main Challenges

* **Extreme sparsity** → cables occupy **< 4%** of the bounding box area
* **Misleading BBoxes** → classified as *Large* (COCO) but mostly empty
* **SAM failure** → background noise overwhelms prompt-based segmentation
* **CNN limitations** → region proposal mechanisms penalize thin structures

### ✅ Why Transformers help

**Global Self-Attention** allows models to:

* capture **long-range dependencies**
* preserve **semantic continuity** of cables
* reconnect visually disjoint cable segments

---

## 📊 Dataset — Deep Dive

### 📦 Key Statistics

| Metric          | Value                |
| --------------- | -------------------- |
| Total images    | **1,242**            |
| Split           | 842 Train / 400 Test |
| Avg fill ratio  | **< 4%**             |
| Median thinness | **1.75 × 10⁻³**      |

### ⚠️ The Dimensional Paradox

```text
Median Mask Area :   608 px
Median BBox Area : 20,740 px
➡️ Bounding boxes are ~34× larger than the actual object
```

👉 This creates a **conceptual mismatch** with standard COCO-style metrics.

---

## 🏗️ Architecture — RF-DETR

**RF-DETR** is a *Transformer-based* detector leveraging:

* **Vision Transformer encoder**
* **Global Self-Attention**
* **Query-based instance prediction**

### ⚙️ Main Configuration

```yaml
encoder: vit_windowed_small
resolution: 1200
num_queries: 200
epochs: 60

lr: 1e-4
lr_encoder: 1e-5
grad_accum_steps: 8
clip_max_norm: 0.1
use_ema: true
ema_decay: 0.999
```

### 🧊 Training Stability

* **EMA (Exponential Moving Average)** → smooths Transformer oscillations
* **Gradient Clipping** → prevents NaN loss
* **Checkpointing** → safe and reproducible training

---

## 🔥 Training Strategy

### 🎯 Differential Learning Rate

* **Decoder**: `1e-4`
* **Backbone**: `1e-5`

👉 The pretrained backbone is updated slowly to preserve generic visual features, while the decoder learns task-specific representations.

### 📈 Gradient Accumulation

* Actual batch size = 1 (VRAM constraint)
* Accumulation steps = 8
* Effective batch size ≈ 8

---

## 🧮 Loss Function

The model jointly minimizes **five loss components**:

[ \mathcal{L} = L_{ce} + L_{bbox} + L_{giou} + L_{mask_ce} + \color{#00f2ff}{L_{mask_dice}} ]

🔑 **Dice Loss** is critical: it ignores the dominant background and emphasizes real object overlap.

---

## 📉 Results

### 📌 Global Metrics

| Metric     | Value     |
| ---------- | --------- |
| **mAP@50** | **0.528** |
| **mAR@10** | **0.280** |

### 📉 Performance vs Scene Density

* 🟢 **0–5 cables** → AP **0.622**, mAR **0.975**
* 🟡 **10–15 cables** → AP **0.199**
* 🔴 **25–50 cables** → AP **0.040**

👉 Mutual occlusion becomes the dominant bottleneck in dense scenes.

---

## 📐 PCA Line Reconstruction (Post-Processing)

For each predicted mask, cable direction is estimated using **PCA-based geometry**:

```python
def pca_line_reconstruct(mask_u8):
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mu = pts.mean(axis=0, keepdims=True)
    X = pts - mu
    C = (X.T @ X) / max(1, n - 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    v = eigvecs[:, np.argmax(eigvals)]
    return recon
```

📌 Even when mask thickness degrades, **angular direction remains accurate** (≈ **0.998° error**).

---

## 🚀 Conclusions

* RF-DETR is **robust** for thin-object segmentation
* Standard detection metrics are **insufficient** for filamentary structures
* **Geometric consistency** (direction) is often more informative than pixel-perfect masks

---

## 🙌 Acknowledgements

> Thank you for your attention!
> We are available for questions or technical discussions.

---

<p align="center">
  <b>Paolo Pangallo & Gianluigi Oricchio</b><br>
  Computer Vision Project — 2026
</p>

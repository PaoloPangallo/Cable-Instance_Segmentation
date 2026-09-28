# Electric Cable Instance Segmentation

**Transformer-based instance segmentation for thin electrical cables in aerial imagery**

This computer-vision project investigates a difficult segmentation setting: objects that are extremely thin, elongated and sparsely represented at pixel level.

The work evaluates **RF-DETR / Transformer-based instance segmentation** and studies why conventional object-detection metrics can be misleading for filamentary structures such as power cables.

## Problem

Electric cables are challenging because:

- they occupy a very small fraction of the image;
- their bounding boxes contain mostly background;
- long segments can be partially occluded;
- pixel-level errors can strongly affect overlap metrics even when cable direction is visually correct.

The dataset used in the project contains **1,242 images** split into 842 training and 400 test images.

## Model

The main experiments use RF-DETR with a Vision Transformer encoder and query-based instance prediction.

A representative training configuration includes:

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

Training uses differential learning rates, gradient accumulation, gradient clipping and exponential moving averages to improve stability.

## Results

Selected global metrics:

| Metric | Value |
| --- | ---: |
| mAP@50 | 0.528 |
| mAR@10 | 0.280 |

Performance decreases substantially as scene density grows, highlighting occlusion and instance separation as the main bottlenecks.

## Geometry-aware post-processing

The project also studies PCA-based line reconstruction from predicted masks.

Even when mask thickness is imperfect, the recovered cable orientation remains highly accurate, with an observed angular error of approximately **0.998°** in the evaluated setting.

This motivates a broader conclusion: for thin filamentary objects, geometric consistency can complement standard overlap-based metrics.

## Tech stack

- Python
- PyTorch
- Computer Vision
- RF-DETR
- Vision Transformers
- Instance Segmentation
- PCA / geometric post-processing

## Setup

Create an isolated environment and install the notebook dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

The trained checkpoint is tracked through **Git LFS** rather than committed as a regular binary blob.

## Repository contents

```text
Cable-Instance_Segmentation/
├── Segmentation.ipynb
├── checkpoint_best_ema.pth
├── requirements.txt
└── README.md
```

The notebook contains the end-to-end experimental workflow.

## Authors

**Paolo Pangallo** and **Gianluigi Oricchio**  
University of Calabria — 2025/2026

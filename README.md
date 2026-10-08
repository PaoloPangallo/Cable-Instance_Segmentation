# Electric Cable Instance Segmentation

**Transformer-based instance segmentation for thin electrical cables in aerial imagery**

This computer-vision project investigates a difficult segmentation setting: objects that are extremely thin, elongated and sparsely represented at pixel level.

The work evaluates **RF-DETR / Transformer-based instance segmentation** and investigates why standard overlap metrics alone may not describe the geometry of long, filamentary objects.

**Motivation:** recognizing “cable pixels” is not enough when several adjacent cables belong to different *instances*. The project studies whether a query-based segmenter can separate them and how geometry and scene density influence evaluation.

### Architecture and project walkthrough

- [**System design** — motivation, model/data flow, experimental branches, two editable Mermaid diagrams and metric caveats](docs/SYSTEM_DESIGN.md)
- [**Guida in italiano** — spiegazione semplice, scelte tecniche, presentazione da colloquio e domande frequenti](docs/PROJECT_WALKTHROUGH_IT.md)

The complete [`Segmentation.ipynb`](Segmentation.ipynb) keeps all **106 original code cells and saved outputs**, organized with Markdown headings. It is a historical research record with multiple dataset versions and experiments, not a reduced demo or a single production-ready Run All pipeline.

## Problem

Electric cables are challenging because:

- they occupy a very small fraction of the image;
- their bounding boxes contain mostly background;
- long segments can be partially occluded;
- pixel-level errors can strongly affect overlap metrics even when cable direction is visually correct.

The README describes a dataset snapshot containing **1,242 images** split into 842 training and 400 test images. The notebook also references additional historical dataset variants; they must not be assumed equivalent.

## Model

The main experiments use RF-DETR with a Vision Transformer encoder and query-based instance prediction.

One representative **historical** training configuration (original notebook code cell 66) includes:

```yaml
encoder: dinov2_windowed_small
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

Selected **archived** results (original notebook code cell 78; not independently rerun in this documentation):

| Metric | Value |
| --- | ---: |
| mAP@50 | 0.528 |
| mAR@10 | 0.280 |

A separate archived analysis finds lower AP50 in scenes containing many annotated cables. Instance separation and occlusion are plausible difficulties, but the available outputs do not isolate their individual causal contributions. Some historical scripts also mislabel COCO AR indices, so their recall values require correction before quantitative reuse.

## Geometry-aware post-processing

The project also studies PCA-based line reconstruction from predicted masks.

**Metric clarification:** one archived evaluation reports `angle_diff ≈ 0.9989555`, but this is a **dimensionless angular similarity score**, calculated by averaging `exp(-0.12 × angular_difference_in_radians)`. It is **not** a mean angular error of 0.998° and should not be reported as such. Actual error in degrees would require a separate computation from the matched directions.

The motivation is to examine **geometric consistency alongside standard COCO mask metrics**, not to claim that orientation alone establishes correct segmentation or instance identity.

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

The notebook preserves the complete historical research workflow, including exploratory PCA/RANSAC/SAM/DeepLSD paths and saved experimental outputs. Some cells modify dataset paths or Python dependencies; inspect their effects before running them.

## Authors

**Paolo Pangallo** and **Gianluigi Oricchio**  
University of Calabria — 2025/2026

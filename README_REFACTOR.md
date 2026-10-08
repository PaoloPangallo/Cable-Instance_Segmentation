# Curated RF-DETR cable segmentation notebook

The original **Segmentation.ipynb** is an experimental archive (106 code cells,
multiple dataset versions/checkpoints, and several PCA/RANSAC/SAM/DeepLSD variants).
This refactor introduces a separate and much shorter analysis workflow without
removing that evidence or rewriting the underlying original experiment.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
jupyter notebook Cable_Segmentation_Refactored.ipynb
```

Edit the notebook's `ProjectConfig` paths: COCO `train/`, `valid/`, `test/`,
prediction JSON paths, checkpoint, report folder. The dataset and complete weights
are not included in these refactoring files. A tracked LFS pointer is **not**
a usable `.pth` file until LFS content is fetched.

The training/inference cells are opt-in and may need RF-DETR API adjustments to
match the **exact historical checkpoint architecture**. The evaluation-only
path loads COCO prediction JSONs. No dataset is modified on import or on
default execution.

## Evaluation protocol

1. COCO dataset read-only audit and density visualisation.
2. Optional training and public `predict()` inference.
3. Confidence threshold chosen with **validation** predictions only.
4. Test AP, AP50, AR1, AR10, AR100 with the selected/frozen threshold.
5. Geometric diagnostics from one-to-one GT-matched instance masks.

With COCO `maxDets=[1,10,100]`, recall at indices 6/7/8 maps to
AR@1, AR@10, AR@100; the historical code mislabels index 6 at least once.
Greedy geometric matching is diagnostic and **not** equivalent to COCOeval.

Original README reported AP50 0.528, AR@10 0.280, and ~0.998° orientation error.
These historical values are **not** recalculated or verified by this refactor.
No claims of a new accuracy improvement are made.

## Limitations

Dataset/checkpoint/predictions were not available together for a GPU run,
so the analysis code can be validated structurally and on synthetic cases
but actual metrics cannot yet be independently reproduced. Do not
combine results across the historical dataset_finale6 / dataset_finale8 /
dataset_finale_aug variants without checking provenance.

Original archive remains unchanged on main. This branch is designed for review.

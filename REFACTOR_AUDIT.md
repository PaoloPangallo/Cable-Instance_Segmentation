# Refactor audit: Cable-Instance_Segmentation

## Inspection
The original notebook contains **106 code cells**, no Markdown cells, and
approximately **32.7 MB** of code and saved experiment outputs. It mixes
multiple dataset versions, model checkpoints, training recipes, and distinct
geometric refinement attempts.

| Original cell (zero-based) | Observation | New treatment |
| --- | --- | --- |
| 1–3 | Extraction cells can delete destination directories | Excluded from run-all |
| 5 | COCO annotations patched in place | Read-only audit; metadata completed in memory |
| 9, 79 | Duplicate polar-line helper | Shared PCA helper with unit tests |
| 12, 66, 67 | Multiple recipes; 66 and 67 duplicated | Representative opt-in training recipe |
| 19, 30 | Test-based confidence threshold sweep | Use **valid** for threshold selection and lock for test |
| 19, 30 | COCO `stats[6]` labeled AR100 | Correct labels AR1=6, AR10=7, AR100=8 |
| 52–58 | SAM refinements mixed with inference and GT checks | Stay in historical archive |
| 80–84 | LDS parameter search and score reranking | Stay exploratory; not canonical |
| 88, 92–94 | External Git clones, environment uninstalls | Excluded from clean workflow |
| 97 | Empty code cell with stored output | Not copied into curated workflow |

## Added
- Dedicated **22-cell** notebook `Cable_Segmentation_Refactored.ipynb`
  with Markdown rationale and disabled-by-default training/inference.
- `cable_seg/core.py`: config, nonmutating inspection, PCA, axial angles,
  one-to-one diagnostic matches and score filtering.
- `cable_seg/coco.py`: optional pycocotools COCO RLE/polygon decoding,
  official AP/AR and geometry diagnostics.
- `cable_seg/inference.py`: optional model.predict -> COCO adapter.
- Unit tests and separate lightweight analysis/training requirements.

## Scope and checks
The original `Segmentation.ipynb` is **untouched** and remains the detailed
experimental history. The curated notebook can run without a dataset and
without destructive setup commands; numeric results cannot be reproduced
until the dataset, prediction files, checkpoint weights and matching package
version are available. Reported historical results (AP50=0.528,
AR@10=0.280, angular error approximately 0.998 degrees) are not recomputed.

A locally assembled equivalent passed **7 Python unit/static tests** and
a no-dataset Jupyter run-all; GitHub-published files have been checked for
presence and notebook JSON structure but no GPU/data execution was performed
on the branch.

"""COCO mask decoding, official segmentation AP/AR, and PCA diagnostics."""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import contextlib
import io
import json
import numpy as np
from .core import axial_angle_error_deg, line_from_mask, match_masks

def _modules():
    try:
        from pycocotools import mask as masks
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("Install pycocotools for COCO evaluation") from exc
    return masks, COCO, COCOeval

def decode_segmentation(segmentation, *, height, width):
    masks, _, _ = _modules()
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=bool)
        rle = masks.merge(masks.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict):
        rle = (masks.frPyObjects(segmentation, height, width)
               if isinstance(segmentation.get("counts"), list) else segmentation)
    else:
        raise ValueError("Expected COCO polygon or RLE")
    out = masks.decode(rle)
    if out.ndim == 3:
        out = np.any(out, axis=2)
    if out.shape != (height, width):
        raise ValueError("Decoded mask size mismatch")
    return out.astype(bool)

def encode_mask(mask):
    masks, _, _ = _modules()
    rle = masks.encode(np.asfortranarray(np.asarray(mask, np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def load_ground_truth(path):
    """Fix optional COCO metadata in memory, never rewrite source annotations."""
    _, COCO, _ = _modules()
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    gt = COCO()
    gt.dataset = deepcopy(source)
    gt.dataset.setdefault("info", {"description": "Cable segmentation", "version": "1.0"})
    gt.dataset.setdefault("licenses", [])
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt

def evaluate_coco(gt_json, predictions, *, max_dets=(1,10,100)):
    """COCO segm stats: AP[0], AP50[1], AR1[6], AR10[7], AR100[8]."""
    _, _, COCOeval = _modules()
    gt = load_ground_truth(gt_json)
    if not predictions:
        return dict.fromkeys(("AP","AP50","AR1","AR10","AR100"), 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions)
        ev = COCOeval(gt, dt, iouType="segm")
        ev.params.maxDets = list(max_dets)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats = ev.stats
    return dict(zip(("AP","AP50","AR1","AR10","AR100"),
                    [float(stats[i]) for i in (0,1,6,7,8)]))

def geometry_diagnostics(gt_json, predictions, *, min_iou=0.5, min_line_pixels=20):
    gt = load_ground_truth(gt_json)
    by_image = {}
    for p in predictions:
        by_image.setdefault(p["image_id"], []).append(p)
    errors = []
    pairs = 0
    for image_id in gt.getImgIds():
        im = gt.imgs[image_id]
        shape = {"height": im["height"], "width": im["width"]}
        anns = gt.loadAnns(gt.getAnnIds(imgIds=[image_id]))
        pred = by_image.get(image_id, [])
        gt_masks = [decode_segmentation(a["segmentation"], **shape) for a in anns]
        pr_masks = [decode_segmentation(p["segmentation"], **shape) for p in pred]
        for i, j, iou in match_masks(gt_masks, pr_masks, min_iou=min_iou):
            pairs += 1
            a = line_from_mask(gt_masks[i], min_pixels=min_line_pixels)
            b = line_from_mask(pr_masks[j], min_pixels=min_line_pixels)
            if a is not None and b is not None:
                errors.append(axial_angle_error_deg(a[1], b[1]))
    return {
        "matched_pairs": pairs, "valid_line_pairs": len(errors),
        "mean_axial_error_deg": float(np.mean(errors)) if errors else None,
        "median_axial_error_deg": float(np.median(errors)) if errors else None,
    }

def save_predictions(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

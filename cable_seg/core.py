"""Side-effect-free COCO summaries, PCA geometry, and mask matching."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np

@dataclass(frozen=True)
class ProjectConfig:
    dataset_root: Path = Path("data/coco")
    checkpoint: Path = Path("checkpoint_best_ema.pth")
    predictions_valid: Path = Path("artifacts/predictions_valid.json")
    predictions_test: Path = Path("artifacts/predictions_test.json")
    output_dir: Path = Path("artifacts/reports")
    seed: int = 42
    score_threshold: float = 0.25
    min_line_pixels: int = 20

    def ann_path(self, split: str) -> Path:
        if split not in ("train", "valid", "test"):
            raise ValueError("Expected train, valid or test")
        return self.dataset_root / split / "_annotations.coco.json"

def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def inspect_coco(annotation: dict) -> dict:
    for key in ("images", "annotations", "categories"):
        if not isinstance(annotation.get(key), list):
            raise ValueError(f"Missing COCO list: {key}")
    images, annotations = annotation["images"], annotation["annotations"]
    ids = [im["id"] for im in images]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs")
    counts = Counter(a["image_id"] for a in annotations)
    arr = np.array([counts.get(i, 0) for i in ids], dtype=int)
    return {
        "images": len(images), "annotations": len(annotations),
        "categories": len(annotation["categories"]),
        "images_without_annotations": int(np.sum(arr == 0)),
        "orphan_annotations": sum(a["image_id"] not in set(ids) for a in annotations),
        "mean_instances_per_image": float(np.mean(arr)) if len(arr) else 0.0,
        "median_instances_per_image": float(np.median(arr)) if len(arr) else 0.0,
        "max_instances_per_image": int(np.max(arr)) if len(arr) else 0,
    }

def instance_density(annotation: dict) -> dict[int, int]:
    counts = Counter(a["image_id"] for a in annotation["annotations"])
    return {im["id"]: counts.get(im["id"], 0) for im in annotation["images"]}

def line_from_mask(mask: np.ndarray, *, min_pixels=20) -> tuple[float, float] | None:
    """PCA normal-form (rho, theta), theta modulo pi, signed rho."""
    if min_pixels < 2:
        raise ValueError("min_pixels must be >= 2")
    ys, xs = np.nonzero(np.asarray(mask) > 0.5)
    if len(xs) < min_pixels:
        return None
    points = np.column_stack([xs, ys]).astype(float)
    mean = points.mean(axis=0)
    p = points - mean
    cov = p.T @ p / len(points)
    eigenvalues, vectors = np.linalg.eigh(cov)
    if eigenvalues[-1] <= 1e-12:
        return None
    direction = vectors[:, -1]
    theta = float((math.atan2(direction[1], direction[0]) + math.pi/2) % math.pi)
    rho = float(mean @ np.array([math.cos(theta), math.sin(theta)]))
    return rho, theta

def axial_angle_error_deg(theta_a, theta_b) -> float:
    d = abs((float(theta_a) - float(theta_b)) % math.pi)
    return math.degrees(min(d, math.pi-d))

def mask_iou(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError("Mask shape mismatch")
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b)/union) if union else 0.0

def match_masks(gt_masks, pred_masks, min_iou=0.5):
    """Greedy one-to-one diagnostic matches, NOT official COCO evaluation."""
    if not 0 <= min_iou <= 1:
        raise ValueError("Invalid IoU threshold")
    candidates = []
    for i, gt in enumerate(gt_masks):
        for j, pred in enumerate(pred_masks):
            iou = mask_iou(gt, pred)
            if iou >= min_iou:
                candidates.append((-iou, i, j))
    used_gt, used_pred, matches = set(), set(), []
    for neg_iou, i, j in sorted(candidates):
        if i not in used_gt and j not in used_pred:
            matches.append((i, j, -neg_iou))
            used_gt.add(i)
            used_pred.add(j)
    return matches

def select_by_score(predictions, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be within [0,1]")
    return [p for p in predictions if float(p.get("score", -1)) >= threshold]

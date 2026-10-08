"""Optional RF-DETR predict adapter, with explicit class-to-COCO mapping."""
from pathlib import Path
import numpy as np
from .coco import encode_mask

def predict_coco(model, annotations, image_dir, *, threshold=0.05,
                 class_index_to_category=None):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    if class_index_to_category is None:
        class_index_to_category = {
            i: c["id"] for i, c in enumerate(
                sorted(annotations["categories"], key=lambda c: c["id"])
            )
        }
    output = []
    for im in annotations["images"]:
        path = Path(image_dir) / im["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        detections = model.predict(str(path), threshold=threshold)
        if detections.mask is None:
            raise ValueError("No segmentation masks returned by RF-DETR")
        masks, scores = detections.mask, detections.confidence
        labels, boxes = detections.class_id, detections.xyxy
        if not (len(masks) == len(scores) == len(labels) == len(boxes)):
            raise ValueError("Inconsistent detection lengths")
        for mask, score, label, bbox in zip(masks, scores, labels, boxes):
            label = int(label)
            if label not in class_index_to_category:
                raise ValueError(f"Unknown model class index: {label}")
            if mask.shape != (im["height"], im["width"]):
                raise ValueError("Output mask/image mismatch")
            if not np.any(mask):
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            output.append({
                "image_id": im["id"],
                "category_id": class_index_to_category[label],
                "segmentation": encode_mask(mask),
                "bbox": [x1, y1, x2-x1, y2-y1],
                "score": float(score),
            })
    return output

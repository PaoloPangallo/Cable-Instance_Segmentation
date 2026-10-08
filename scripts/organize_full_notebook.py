"""Organize the complete ORIGINAL Segmentation.ipynb, without dropping code or outputs.

This is a non-destructive, idempotent formatting pass. Original code order,
execution counts and saved plots/metrics are preserved exactly.
"""
import copy
import json
from pathlib import Path

MARKER = "<!-- original-106-cells-preserved -->"

SECTIONS = {
    0: ("01 | Environment and data preparation",
        "RF-DETR installation, dataset extraction and file copying. **Caution:** some original cells delete or overwrite directories. Verify Colab and Drive paths before execution."),
    5: ("02 | COCO dataset exploration",
        "COCO sanity checks, bounding-box statistics, mask geometry and split consistency. An original cell may write missing JSON metadata back to disk."),
    11: ("03 | RF-DETR setup and PCA reconstruction",
        "First training configuration, geometric ground-truth reconstruction and exploratory visualization."),
    16: ("04 | Dataset filtering and initial evaluation",
        "Dense-image filtering, COCO evaluation helpers, RF-DETR inference and GT/prediction diagnostics."),
    23: ("05 | Dataset morphology and fragmentation",
        "Train/test image counts, connected components, mask shape statistics and geometric comparisons."),
    30: ("06 | RF-DETR inference and mask repair experiments",
        "Several alternative attempts with thresholds, PCA, RANSAC and mask repair. These are experiments, not a single mandatory run-all pipeline."),
    47: ("07 | COCO metric diagnostics and SAM experiments",
        "Oracle/model diagnostics, hard images, instance matching and multiple SAM refinement candidates."),
    60: ("08 | Dataset variants and alternative training",
        "Split comparisons, dense-instance oversampling and alternative training recipes. All historical configurations are retained."),
    72: ("09 | Image inspection and COCO matching",
        "Visual sanity checks, per-image masks and examination of instance-level matching."),
    80: ("10 | Polar representation and line-detection scoring",
         "Line parameters, ranking and score optimisation. Ground-truth-guided parameter selection is exploratory and not an independent test evaluation."),
    85: ("11 | Geometry-aware COCO post-processing",
         "Mask-NMS, line diversification and proposed ranking/merging refinements with saved experiment outputs."),
    88: ("12 | DeepLSD experiments",
         "Optional external tool setup and line-guided inference/evaluation. Extra dependencies may be required."),
    92: ("13 | External baselines and environment experiments",
         "**Caution:** these cells clone other repositories and may uninstall PyTorch/Ultralytics. They are not required to run the preceding experiments."),
    96: ("14 | Further line-guided refinements",
         "Additional saved mask ranking and geometry-based post-processing experiments."),
    99: ("15 | Failure analysis",
         "Density-stratified metrics, false positives/negatives, thickness/length distributions and mask IoU."),
}

NOTES = {
    67: "**Repeated training configuration.** This original cell is intentionally retained along with its saved output; no historical experiment was deleted.",
    79: "**Repeated polar-coordinate helper.** Kept exactly where it appeared in the source notebook.",
    97: "**Output without source.** This original cell has no code, but contains three saved outputs. They are preserved.",
}

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}

def src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s

def organize(nb):
    if nb["cells"] and MARKER in src(nb["cells"][0]):
        return nb
    original = nb["cells"]
    if len(original) != 106 or any(c["cell_type"] != "code" for c in original):
        raise ValueError("Expected exactly the 106 original code cells. Refusing rewrite.")
    intro = """# Electric Cable Instance Segmentation

<!-- original-106-cells-preserved -->

**RF-DETR, COCO evaluation, PCA, RANSAC, SAM and line-guided post-processing.**

This is the **complete original research notebook, organized for reading**.
All 106 code cells, in their original order, and all saved output cells remain.
Nothing has been replaced with a shorter demo or rewritten training pipeline.

## Contents

""" + "\n".join(
        f"- **{title}**" for title, _ in SECTIONS.values()
    ) + """

> **Execution warning:** the cells document multiple historical dataset versions,
> checkpoints and environments. Some modify datasets or dependencies. Treat the
> notebook as an experimental record; inspect paths before running individual
> sections, and do not blindly select Run All. Saved metrics are historical
> outputs, not newly reproduced measurements.
"""
    result = copy.deepcopy(nb)
    result["cells"] = [md(intro)]
    for index, cell in enumerate(original):
        if index in SECTIONS:
            title, description = SECTIONS[index]
            result["cells"].append(md(f"## {title}\n\n{description}"))
        if index in NOTES:
            result["cells"].append(md(NOTES[index]))
        current = copy.deepcopy(cell)
        for output in current.get("outputs", []):
            if output.get("output_type") == "stream":
                # Some Colab exports attach illegal metadata to stream outputs.
                # Removing this invalid wrapper preserves the full output text.
                output.pop("metadata", None)
        result["cells"].append(current)
    original_codes = [src(c) for c in original]
    produced_codes = [src(c) for c in result["cells"] if c["cell_type"] == "code"]
    assert produced_codes == original_codes, "Original code must not change."
    original_outputs = [o for c in original for o in c.get("outputs", [])]
    new_outputs = [o for c in result["cells"] for o in c.get("outputs", [])]
    assert len(original_outputs) == len(new_outputs) == 446, "Missing original outputs."
    for old, new in zip(original_outputs, new_outputs):
        before = copy.deepcopy(old)
        if before.get("output_type") == "stream":
            before.pop("metadata", None)
        assert before == new, "Saved output changed."
    return result

def main():
    path = Path("Segmentation.ipynb")
    original = json.loads(path.read_text(encoding="utf-8"))
    organized = organize(original)
    import nbformat
    nbformat.validate(organized)
    if organized != original:
        path.write_text(json.dumps(organized, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cells = organized["cells"]
    assert sum(c["cell_type"] == "code" for c in cells) == 106
    print("PASS: 106 original code cells, 446 saved outputs, same code order; notebook valid.")
    print("Markdown headings:", sum(c["cell_type"] == "markdown" for c in cells))

if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import supervision as sv
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

from microvision.config import OBJECT_CLASSES

# Configuration
CONF_THRESH = 0.5
IOU_THRESH = 0.6

GT_JSON = "data/eval/gt.json"
OUT_DIR = Path("results/metrics")
OUT_DIR.mkdir(exist_ok=True)

# Map your model names to their prediction files
MODELS = {
    "RF-DETR": "data/eval/rfdetr.json",
    "YOLO11": "data/eval/yolo11.json",
    "Faster R-CNN": "data/eval/fasterrcnn.json",
}

CLASS_NAMES = [item.capitalize() for item in OBJECT_CLASSES.values()]

# Load ground truth
print("Loading ground truth...")
coco = COCO(GT_JSON)
gt_map = {}

for img in coco.dataset["images"]:
    img_id = img["id"]
    anns = coco.imgToAnns.get(img_id, [])

    # Extract annotations and convert to Supervision format
    boxes, classes = [], []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        classes.append(ann["category_id"] - 1)  # Ensure 0-indexed

    boxes = np.array(boxes, dtype=float).reshape(-1, 4) if boxes else np.empty((0, 4))
    classes = np.array(classes, dtype=int) if classes else np.empty((0,))

    gt_map[img_id] = sv.Detections(xyxy=boxes, class_id=classes)

# Evaluation
all_results = []

# Prepare the plot figure
fig, axes = plt.subplots(1, 3, figsize=(19, 5), dpi=600)
plt.subplots_adjust(wspace=0.5)

for idx, (model_name, pred_path) in enumerate(MODELS.items()):
    print(f"\nEvaluating {model_name}...")

    # Load Predictions
    with open(pred_path) as f:
        raw_preds = json.load(f)

    # Filter by confidence and group by image
    pred_map_raw = {}
    for det in raw_preds:
        if det["score"] < CONF_THRESH:
            continue
        img_id = det["image_id"]
        x, y, w, h = det["bbox"]

        if img_id not in pred_map_raw:
            pred_map_raw[img_id] = {"boxes": [], "classes": [], "scores": []}

        pred_map_raw[img_id]["boxes"].append([x, y, x + w, y + h])
        pred_map_raw[img_id]["classes"].append(det["category_id"] - 1)
        pred_map_raw[img_id]["scores"].append(det["score"])

    # Convert to Supervision format
    pred_map = {}
    for img_id, data in pred_map_raw.items():
        pred_map[img_id] = sv.Detections(
            xyxy=np.array(data["boxes"]),
            class_id=np.array(data["classes"]),
            confidence=np.array(data["scores"]),
        )

    # Align inputs for confusion matrix
    targets = []
    predictions = []
    for img_id in gt_map.keys():
        targets.append(gt_map[img_id])
        predictions.append(pred_map.get(img_id, sv.Detections.empty()))

    # Generate Confusion Matrix
    cm = sv.ConfusionMatrix.from_detections(
        predictions=predictions,
        targets=targets,
        classes=CLASS_NAMES,
        conf_threshold=CONF_THRESH,
        iou_threshold=IOU_THRESH,
    )

    # Extract the matrix data (numpy array)
    matrix_data = cm.matrix

    # Plot using seaborn for better control
    sns.heatmap(
        matrix_data.astype(int),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[idx],
        xticklabels=CLASS_NAMES + ["Background"],
        yticklabels=CLASS_NAMES + ["Background"],
        cbar=False,  # Remove colorbar to save space
    )

    axes[idx].set_title(model_name)
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Annotated")

    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90, ha="center")
    axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0, ha="right")

    # Extract Metrics from the Matrix
    matrix = cm.matrix
    model_tp = model_fp = model_fn = 0

    for i, class_name in enumerate(CLASS_NAMES):
        # In supervision CM: Rows = Targets (GT), Columns = Predictions
        TP = matrix[i, i]
        FN = np.sum(matrix[i, :]) - TP
        FP = np.sum(matrix[:, i]) - TP

        # Accumulate for 'All Classes'
        model_tp += TP
        model_fn += FN
        model_fp += FP

        # Calculate class-specific metrics
        accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        all_results.append(
            {
                "Class": class_name,
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
            }
        )

    # Calculate 'All Classes' metrics
    acc_all = (
        model_tp / (model_tp + model_fp + model_fn)
        if (model_tp + model_fp + model_fn) > 0
        else 0
    )
    p_all = model_tp / (model_tp + model_fp) if (model_tp + model_fp) > 0 else 0
    r_all = model_tp / (model_tp + model_fn) if (model_tp + model_fn) > 0 else 0
    f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0

    all_results.append(
        {
            "Class": "All Classes",
            "Model": model_name,
            "Accuracy": acc_all,
            "Precision": p_all,
            "Recall": r_all,
            "F1-Score": f1_all,
        }
    )

plt.savefig(OUT_DIR / "combined_confusion_matrices.png", bbox_inches="tight")

# Format results
# Ensure the DataFrame is sorted by Class to group them
df = pd.DataFrame(all_results)
df = df.sort_values(
    by=["Class", "Model"],
    key=lambda x: x.map(
        {
            "Pedestrian": 0,
            "Bicycle": 1,
            "Cyclist": 2,
            "E-scooter": 3,
            "E-scooterist": 4,
            "All Classes": 5,
        }
        if x.name == "Class"
        else {model: i for i, model in enumerate(MODELS.keys())}
    ),
)

# Set MultiIndex to group by Class and then Model
df = df.set_index(["Class", "Model"])

# Format values to 3 decimal places
df_formatted = df.map(lambda x: f"{x:.3f}")

# Display
print("\nMetrics Table:\n")
print(df_formatted.to_string())

# Save for your report
df_formatted.to_csv(OUT_DIR / "metrics_table_grouped.csv")


def generate_latex_table(df):
    """Generates a LaTeX table string"""
    latex = [
        "\\begin{table}[h]",
        "\\caption{Comparison of models on the test set.}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\textbf{Class} & \\textbf{Model} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} \\\\",
        "\\midrule",
    ]

    classes = df.index.get_level_values("Class").unique()

    for cls in classes:
        cls_df = df.loc[cls]

        # Add multirow for the class name
        latex.append(f"\\multirow{{3}}{{*}}{{{cls}}}")

        for model in cls_df.index:
            row = cls_df.loc[model]

            # Find max in each column to bold it
            vals = row.astype(float)
            row_str = []
            for col in df.columns:
                val = vals[col]
                # Compare against other models for this specific class and metric
                is_max = val == cls_df[col].astype(float).max()
                val_str = f"\\textbf{{{val:.3f}}}" if is_max else f"{val:.3f}"
                row_str.append(val_str)

            latex.append(f" & {model} & " + " & ".join(row_str) + " \\\\")

        latex.append("\\midrule")

    latex[-1] = "\\bottomrule"  # Replace last midrule with bottomrule
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    return "\n".join(latex)


if __name__ == "__main__":
    # Run the generator
    latex_code = generate_latex_table(df)
    print("\nLaTeX table:\n")
    print(latex_code)

    # Save to file
    with open(OUT_DIR / "table.tex", "w") as f:
        f.write(latex_code)

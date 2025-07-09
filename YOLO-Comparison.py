import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ---- CONFIG ----
gt_folder = "comparison-data/ground_truth/ground_truth_combined/grouped"
plain_folder = "comparison-data/detections_no_seg/grouped"
seg_folder = "comparison-data/detections_seg"
output_folder = "comparison-data/results"
os.makedirs(output_folder, exist_ok=True)

sns.set_theme(style="whitegrid")

# ---- CLASS MAPPING ----
def map_to_group(cls):
    if cls in [0, 1, 2]: return 0  # VRU
    elif cls in [3, 4]: return 1  # Fast
    elif cls in [5, 6]: return 2  # Slow
    else: return -1

# ---- NEAREST MATCHING FUNCTION ----
def match_predictions(df_gt, df_pred):
    matched_gt, matched_pred, conf_scores = [], [], []

    for frame in df_gt["frame"].unique():
        gt_frame = df_gt[df_gt["frame"] == frame]
        pred_frame = df_pred[df_pred["frame"] == frame].copy()

        for _, gt_row in gt_frame.iterrows():
            pred_frame["dist"] = ((pred_frame["x_center"] - gt_row["x_center"])**2 +
                                  (pred_frame["y_center"] - gt_row["y_center"])**2) ** 0.5
            nearest = pred_frame.sort_values("dist").head(1)
            if not nearest.empty:
                matched_gt.append(gt_row["group_class"])
                matched_pred.append(nearest["group_class"].values[0])
                conf_scores.append(nearest.get("confidence", 1.0))

    return matched_gt, matched_pred, conf_scores

# ---- MAIN LOOP ----
labels = ["VRU", "Fast", "Slow"]
int_labels = [0, 1, 2]

gt_files = sorted(glob.glob(os.path.join(gt_folder, "*_grouped_f0to200.csv")))

for gt_file in gt_files:
    base = os.path.basename(gt_file).replace("_combined_grouped_f0to200.csv", "")
    try:
        plain_file = os.path.join(plain_folder, f"{base}_detections1_grouped_f0to200.csv")
        seg_file = os.path.join(seg_folder, f"{base}_detections.csv")

        # Load data
        df_gt = pd.read_csv(gt_file)
        df_plain = pd.read_csv(plain_file)
        df_seg = pd.read_csv(seg_file)

        # Preprocess predictions
        df_seg["group_class"] = df_seg["class"].apply(map_to_group)
        df_plain = df_plain[df_plain["group_class"] != -1]
        df_seg = df_seg[df_seg["group_class"] != -1]

        # Match predictions
        gt_p, pred_p, _ = match_predictions(df_gt, df_plain)
        gt_s, pred_s, _ = match_predictions(df_gt, df_seg)

        # Compute metrics
        metrics_df = pd.DataFrame({
            "Model": ["YOLOv8 Only", "YOLOv8 + Segmentation"],
            "F1 Score": [
                f1_score(gt_p, pred_p, labels=int_labels, average="macro", zero_division=0),
                f1_score(gt_s, pred_s, labels=int_labels, average="macro", zero_division=0)
            ],
            "Precision": [
                precision_score(gt_p, pred_p, labels=int_labels, average="macro", zero_division=0),
                precision_score(gt_s, pred_s, labels=int_labels, average="macro", zero_division=0)
            ],
            "Recall": [
                recall_score(gt_p, pred_p, labels=int_labels, average="macro", zero_division=0),
                recall_score(gt_s, pred_s, labels=int_labels, average="macro", zero_division=0)
            ]
        })

        # Confusion matrices
        cm_p = confusion_matrix(gt_p, pred_p, labels=int_labels)
        cm_s = confusion_matrix(gt_s, pred_s, labels=int_labels)

        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        sns.heatmap(cm_p, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axs[0])
        axs[0].set_title(f"{base} - YOLOv8 Only")
        axs[0].set_xlabel("Predicted")
        axs[0].set_ylabel("Actual")

        sns.heatmap(cm_s, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axs[1])
        axs[1].set_title(f"{base} - YOLOv8 + Segmentation")
        axs[1].set_xlabel("Predicted")
        axs[1].set_ylabel("Actual")

        sns.barplot(data=metrics_df.melt(id_vars="Model"), x="variable", y="value", hue="Model", ax=axs[2])
        axs[2].set_title("F1, Precision, Recall")
        axs[2].set_ylim(0, 1.05)
        axs[2].set_ylabel("Score")

        plt.tight_layout()
        output_path = os.path.join(output_folder, f"{base}_analysis.png")
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved plot: {output_path}")

    except Exception as e:
        print(f"❌ Error processing {base}: {e}")
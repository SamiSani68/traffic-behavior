# python segmentation/F1.py \
#   --gt_folder segmentation/comparison-data-segmentation/ground_truth/ground_truth_combined/grouped \
#   --plain_folder segmentation/comparison-data-segmentation/detections_no_seg/grouped \
#   --seg_folder segmentation/comparison-data-segmentation/detections_seg \
#   --output_folder segmentation/comparison-data-segmentation/results
#   [--average weighted|macro|micro]   # default: weighted

import os, re, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import argparse

sns.set_theme(style="whitegrid")

def map_to_group(cls):
    if cls in [0,1,2]: return 0  # VRU
    if cls in [3,4]:   return 1  # Fast
    if cls in [5,6]:   return 2  # Slow
    return -1

def match_predictions(df_gt, df_pred):
    """
    Greedy NN matching per frame in (x_center, y_center). Each prediction
    is used at most once (dropped after matching).
    """
    matched_gt, matched_pred, conf_scores = [], [], []
    for frame in df_gt["frame"].unique():
        gt_frame = df_gt[df_gt["frame"] == frame]
        pred_frame = df_pred[df_pred["frame"] == frame].copy()
        for _, gt_row in gt_frame.iterrows():
            if pred_frame.empty:
                continue
            pred_frame["dist"] = ((pred_frame["x_center"]-gt_row["x_center"])**2 +
                                  (pred_frame["y_center"]-gt_row["y_center"])**2) ** 0.5
            nearest = pred_frame.sort_values("dist").head(1)
            if not nearest.empty:
                matched_gt.append(gt_row["group_class"])
                matched_pred.append(nearest["group_class"].values[0])
                conf_scores.append(
                    nearest["confidence"].values[0] if "confidence" in nearest else 1.0
                )
                # drop the used prediction so it can't match again
                pred_frame = pred_frame.drop(nearest.index)
    return matched_gt, matched_pred, conf_scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_folder", default="comparison-data/ground_truth/ground_truth_combined/grouped")
    ap.add_argument("--plain_folder", default="comparison-data/detections_no_seg/grouped")
    ap.add_argument("--seg_folder", default="comparison-data/detections_seg")
    ap.add_argument("--output_folder", default="comparison-data/results")
    ap.add_argument("--average", choices=["weighted","macro","micro"], default="weighted",
                    help="Averaging for P/R/F1 (default: weighted)")
    args = ap.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    gt_files = sorted(glob.glob(os.path.join(args.gt_folder, "*_grouped_f0to200.csv")))
    if not gt_files:
        gt_files = sorted(glob.glob(os.path.join(args.gt_folder, "*_combined_grouped_f0to200.csv")))

    print(f"[INFO] GT folder: {os.path.abspath(args.gt_folder)}")
    print(f"[INFO] Found {len(gt_files)} GT files")
    if gt_files[:3]:
        print("[INFO] Examples:", [os.path.basename(f) for f in gt_files[:3]])

    labels = ["VRU","Fast","Slow"]
    int_labels = [0,1,2]
    summary_rows = []
    avg = args.average  # weighted by default

    for gt_file in gt_files:
        fn = os.path.basename(gt_file)
        base = re.sub(r"_(combined_)?grouped_f0to200\.csv$", "", fn)

        plain_file = os.path.join(args.plain_folder, f"{base}_detections1_grouped_f0to200.csv")
        seg_file   = os.path.join(args.seg_folder,   f"{base}_detections.csv")

        if not os.path.exists(plain_file):
            matches = glob.glob(os.path.join(args.plain_folder, f"{base}*grouped_f0to200.csv"))
            if matches:
                plain_file = matches[0]
        if not os.path.exists(seg_file):
            matches = glob.glob(os.path.join(args.seg_folder, f"{base}*detections*.csv"))
            if matches:
                seg_file = matches[0]

        if not (os.path.exists(plain_file) and os.path.exists(seg_file)):
            print(f"[WARN] Skipping {base} (missing files)")
            print(f"       plain: {plain_file if os.path.exists(plain_file) else 'NOT FOUND'}")
            print(f"       seg:   {seg_file   if os.path.exists(seg_file)   else 'NOT FOUND'}")
            continue

        try:
            df_gt    = pd.read_csv(gt_file)
            df_plain = pd.read_csv(plain_file)
            df_seg   = pd.read_csv(seg_file)

            # Ensure grouped labels
            if "group_class" not in df_gt.columns and "class" in df_gt.columns:
                df_gt["group_class"] = df_gt["class"].apply(map_to_group)
            if "group_class" not in df_plain.columns and "class" in df_plain.columns:
                df_plain["group_class"] = df_plain["class"].apply(map_to_group)
            if "group_class" not in df_seg.columns and "class" in df_seg.columns:
                df_seg["group_class"] = df_seg["class"].apply(map_to_group)

            # Filter valid classes
            df_gt    = df_gt[df_gt["group_class"].isin(int_labels)]
            df_plain = df_plain[df_plain["group_class"].isin(int_labels)]
            df_seg   = df_seg[df_seg["group_class"].isin(int_labels)]

            required = {"frame","x_center","y_center","group_class"}
            for name, df in [("GT",df_gt),("YOLO",df_plain),("YOLO+Seg",df_seg)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            gt_p, pred_p, _ = match_predictions(df_gt, df_plain)
            gt_s, pred_s, _ = match_predictions(df_gt, df_seg)

            # --- Weighted (default) metrics ---
            f1_p  = f1_score(gt_p,  pred_p,  labels=int_labels, average=avg, zero_division=0)
            pre_p = precision_score(gt_p, pred_p, labels=int_labels, average=avg, zero_division=0)
            rec_p = recall_score(gt_p, pred_p, labels=int_labels, average=avg, zero_division=0)

            f1_s  = f1_score(gt_s,  pred_s,  labels=int_labels, average=avg, zero_division=0)
            pre_s = precision_score(gt_s, pred_s, labels=int_labels, average=avg, zero_division=0)
            rec_s = recall_score(gt_s, pred_s, labels=int_labels, average=avg, zero_division=0)

            summary_rows.append({
                "scene": base,
                "F1_YOLO_only": f1_p, "Precision_YOLO_only": pre_p, "Recall_YOLO_only": rec_p,
                "F1_YOLO_seg":  f1_s, "Precision_YOLO_seg":  pre_s, "Recall_YOLO_seg":  rec_s,
                "n_matches_yolo": len(gt_p), "n_matches_seg": len(gt_s)
            })

            cm_p = confusion_matrix(gt_p, pred_p, labels=int_labels)
            cm_s = confusion_matrix(gt_s, pred_s, labels=int_labels)

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            # Confusion matrices
            sns.heatmap(cm_p, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels, ax=axs[0])
            axs[0].set_title(f"{base} - YOLO Only")
            axs[0].set_xlabel("Predicted"); axs[0].set_ylabel("Actual")

            sns.heatmap(cm_s, annot=True, fmt="d", cmap="Greens",
                        xticklabels=labels, yticklabels=labels, ax=axs[1])
            axs[1].set_title(f"{base} - YOLO + Seg")
            axs[1].set_xlabel("Predicted"); axs[1].set_ylabel("Actual")

            # Bar chart of metrics
            metrics_df = pd.DataFrame({
                "Metric": ["F1","Precision","Recall"],
                "YOLO Only": [f1_p, pre_p, rec_p],
                "YOLO + Seg": [f1_s, pre_s, rec_s]
            })
            m_long = metrics_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
            ax = axs[2]
            sns.barplot(data=m_long, x="Metric", y="Score", hue="Model", ax=ax)
            ax.set_title(f"{avg.capitalize()} scores")
            ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
            for p in ax.patches:
                val = p.get_height()
                ax.annotate(f"{val:.2f}", (p.get_x()+p.get_width()/2., val),
                            ha="center", va="bottom", fontsize=10, xytext=(0,3), textcoords="offset points")

            plt.tight_layout()
            out_png = os.path.join(args.output_folder, f"{base}_analysis.png")
            plt.savefig(out_png); plt.close()

            print(f"[OK] {base}  {avg.capitalize()} F1 — YOLO: {f1_p:.3f} | YOLO+Seg: {f1_s:.3f}")

        except Exception as e:
            print(f"[ERR] {base}: {e}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        per_scene_csv = os.path.join(args.output_folder, "metrics_summary.csv")
        summary_df.to_csv(per_scene_csv, index=False)
        print(f"[SAVE] Per-scene metrics: {per_scene_csv}")

        overall = summary_df[
            ["F1_YOLO_only","Precision_YOLO_only","Recall_YOLO_only",
             "F1_YOLO_seg","Precision_YOLO_seg","Recall_YOLO_seg"]
        ].mean().to_frame(name="Mean").round(4)
        overall_csv = os.path.join(args.output_folder, "metrics_summary_overall.csv")
        overall.to_csv(overall_csv)
        print("[MEAN]\n", overall)
        print(f"[SAVE] Overall means: {overall_csv}")
    else:
        print("No scenes processed; no summary written.")

if __name__ == "__main__":
    main()
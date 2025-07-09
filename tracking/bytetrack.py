import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import Namespace
from yolox.tracker.byte_tracker import BYTETracker
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# CONFIG
LOG_DIR = "video-analysis/results/logs"
VIDEO_DIR = "video-analysis/videos"
OUT_DIR = "video-analysis/tracked_videos"
PLOT_DIR = "video-analysis/analysis_plots"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Class mapping
def map_class(c):
    if c in [0, 1, 2]: return 0  # VRU
    elif c in [3, 4]: return 1  # Fast
    elif c in [5, 6]: return 2  # Slow
    return -1

group_names = ["VRU", "Fast", "Slow"]
class_map = {0: "VRU", 1: "Fast", 2: "Slow", -1: "Unknown"}
palette = {"VRU": "green", "Fast": "red", "Slow": "blue", "Unknown": "gray"}

# Tracker configuration
tracker_args = Namespace(
    track_thresh=0.3,
    match_thresh=0.8,
    track_buffer=30,
    frame_rate=30,
    mot20=False
)

for csv_file in sorted(Path(LOG_DIR).glob("*.csv")):
    video_base = csv_file.stem.replace("_detections", "")
    video_path = os.path.join(VIDEO_DIR, f"{video_base}.MP4")
    video_out_path = os.path.join(OUT_DIR, f"{video_base}_tracked.mp4")
    csv_out_path = os.path.join(OUT_DIR, f"{video_base}_tracks.csv")

    if not os.path.exists(video_path):
        print(f"Missing video: {video_path}")
        continue

    print(f"Processing {video_base}")
    df = pd.read_csv(csv_file)
    df["group_class"] = df["class"].apply(map_class)

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    tracker = BYTETracker(tracker_args)
    frame_id = 0
    w_box, h_box = 50, 50
    all_tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = df[df["frame"] == frame_id]
        det_list = []

        for _, row in frame_dets.iterrows():
            x1 = row["x_center"] - w_box / 2
            y1 = row["y_center"] - h_box / 2
            x2 = x1 + w_box
            y2 = y1 + h_box
            det_list.append([x1, y1, x2, y2, row["confidence"], row["group_class"]])

        if det_list:
            det_array = np.array(det_list)
            online_targets = tracker.update(det_array, (h, w), (h, w))
        else:
            online_targets = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            cls = int(getattr(t, "cls", -1))
            x, y, w_box, h_box = map(int, tlwh)
            x_center = x + w_box / 2
            y_center = y + h_box / 2

            if 0 <= cls < len(group_names):
                label = f"{group_names[cls]}-{tid}"
                color = (0, 255, 0) if cls == 0 else (0, 0, 255) if cls == 1 else (255, 0, 0)
            else:
                label = f"Unknown-{tid}"
                color = (128, 128, 128)

            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            all_tracks.append({
                "frame": frame_id,
                "track_id": tid,
                "class": cls,
                "x_center": x_center,
                "y_center": y_center,
                "width": w_box,
                "height": h_box
            })

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    df_out = pd.DataFrame(all_tracks)
    df_out.to_csv(csv_out_path, index=False)
    print(f"Video saved: {video_out_path}")
    print(f"CSV saved:   {csv_out_path}")

    # Generate plots
    df_out["class_label"] = df_out["class"].map(class_map)

    # Plot 1: class distribution
    counts = df_out["class_label"].value_counts().reindex(class_map.values()).fillna(0)
    counts.plot(kind="bar", color=[palette[k] for k in counts.index])
    plt.title(f"Track Count per Class - {video_base}")
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{video_base}_class_count.png"))
    plt.close()

    # Plot 2: per-frame class count
    frame_counts = df_out.groupby(["frame", "class_label"]).size().unstack(fill_value=0)
    frame_counts.plot(color=[palette[k] for k in frame_counts.columns])
    plt.title(f"Objects Tracked per Frame - {video_base}")
    plt.xlabel("Frame")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{video_base}_per_frame_count.png"))
    plt.close()

    # Plot 3: Heatmap
    known_df = df_out[df_out["class"] != -1]
    if not known_df.empty:
        sns.kdeplot(data=known_df, x="x_center", y="y_center", fill=True, cmap="viridis", bw_adjust=0.6)
        plt.title(f"Heatmap of Object Locations - {video_base}")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{video_base}_heatmap.png"))
        plt.close()

    # Plot 4: Scatter
    sns.scatterplot(data=df_out, x="x_center", y="y_center", hue="class_label",
                    palette=palette, alpha=0.5, linewidth=0)
    plt.title(f"Scatter of Tracked Locations - {video_base}")
    plt.xlabel("X Center")
    plt.ylabel("Y Center")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{video_base}_scatter.png"))
    plt.close()

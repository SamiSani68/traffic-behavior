#python tracking/deepsort-analyse.py
#input:video-analysis/tracked_videos/deepsort
#output:tracking/deepsort_analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

csv_dir = Path("video-analysis/tracked_videos/deepsort")
output_dir = Path("tracking/deepsort_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

class_map = {0: "VRU", 1: "Fast", 2: "Slow", -1: "Unknown"}
palette = {"VRU": "green", "Fast": "red", "Slow": "blue", "Unknown": "gray"}

csv_files = sorted(csv_dir.glob("*_deepsort_tracks.csv"))

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    base = csv_file.stem.replace("_deepsort_tracks", "")
    print(f"Analyzing: {base}")

    if df.empty or "track_id" not in df.columns or "frame" not in df.columns:
        print(f"Skipping {base}: Invalid or empty data.")
        continue

    df["class_label"] = df["class"].map(class_map)

    counts = df["class_label"].value_counts().reindex(class_map.values()).fillna(0)
    counts.plot(kind="bar", color=[palette[k] for k in counts.index])
    plt.title(f"Track Count per Class - {base}")
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / f"{base}_class_count.png")
    plt.close()

    frame_counts = df.groupby(["frame", "class_label"]).size().unstack(fill_value=0)
    frame_counts.plot(color=[palette[k] for k in frame_counts.columns])
    plt.title(f"Objects Tracked per Frame - {base}")
    plt.xlabel("Frame")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base}_per_frame_count.png")
    plt.close()

    known_df = df[df["class"] != -1]
    if not known_df.empty:
        sns.kdeplot(data=known_df, x="x_center", y="y_center", fill=True, cmap="viridis", bw_adjust=0.6)
        plt.title(f"Heatmap of Object Locations - {base}")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(output_dir / f"{base}_heatmap.png")
        plt.close()

    sns.scatterplot(data=df, x="x_center", y="y_center", hue="class_label",
                    palette=palette, alpha=0.5, linewidth=0)
    plt.title(f"Scatter of Tracked Locations - {base}")
    plt.xlabel("X Center")
    plt.ylabel("Y Center")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{base}_scatter.png")
    plt.close()

print("DeepSORT plots saved to:", output_dir)
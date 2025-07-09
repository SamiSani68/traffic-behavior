import pandas as pd
from pathlib import Path
import math

# === Directory configuration ===
SUMMARY_CSV = Path("behavior/arc_point_summary.csv")
DEEPSORT_DIR = Path("video-analysis/tracked_videos/deepsort")
OUTPUT_DIR = Path("behavior/projected_tracks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Coordinate projection function ===
def project_to_common_plane(x, y, pixels_per_meter, angle_deg_horizontal):
    x_m = x / pixels_per_meter
    y_m = y / pixels_per_meter
    theta = math.radians(angle_deg_horizontal)
    x_ground = x_m * math.cos(theta) + y_m * math.sin(theta)
    y_ground = -x_m * math.sin(theta) + y_m * math.cos(theta)
    return x_ground, y_ground

# === Load summary CSV ===
summary_df = pd.read_csv(SUMMARY_CSV)

# === Process all *_deepsort_tracks.csv files ===
for track_file in sorted(DEEPSORT_DIR.glob("*_deepsort_tracks.csv")):
    base_name = track_file.stem.replace("_deepsort_tracks", "")
    match_row = summary_df[summary_df["file"].str.contains(base_name)]

    if match_row.empty:
        print(f" No matching summary found for: {base_name}")
        continue

    # Extract parameters
    pixels_per_meter = match_row["pixels_per_meter"].values[0]
    angle_deg = match_row["angle_deg_horizontal"].values[0]

    # Load tracking data
    df = pd.read_csv(track_file)

    # Apply projection
    x_ground, y_ground = project_to_common_plane(
        df["x_center"].to_numpy(), df["y_center"].to_numpy(),
        pixels_per_meter, angle_deg
    )

    df["x_ground"] = x_ground
    df["y_ground"] = y_ground

    # Save result
    output_path = OUTPUT_DIR / f"{base_name}_projected.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path.name}")

print("All DeepSORT coordinates projected to ground plane.")

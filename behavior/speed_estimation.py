import pandas as pd
import numpy as np
import json
from pathlib import Path

# === Configuration ===
TRACKS_DIR = Path("video-analysis/tracked_videos/deepsort")
CALIB_JSON = Path("behavior/GCP_points_yC/dynamic_meters_per_pixel_axis_map.json")
OUTPUT_DIR = TRACKS_DIR / "speed_estimated_csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the calibration JSON
with open(CALIB_JSON, "r") as f:
    calibration_map = json.load(f)

def interpolate_scale(position, points):
    """Interpolate with clamping to avoid extrapolation"""
    pos_list, mpp_list = zip(*points)
    position = max(min(position, max(pos_list)), min(pos_list))
    return np.interp(position, pos_list, mpp_list)

# Process each DeepSORT CSV
for csv_file in TRACKS_DIR.glob("*_deepsort_tracks.csv"):
    video_id = csv_file.stem.replace("_deepsort_tracks", "")
    if video_id not in calibration_map:
        print(f"Skipping {video_id}: No calibration data.")
        continue

    axis, calibration_points = calibration_map[video_id]
    df = pd.read_csv(csv_file).sort_values(by=["track_id", "frame"])

    speed_mps, speed_kmph = [], []

    # Compute speed track-by-track
    for _, track in df.groupby("track_id"):
        prev_row = None
        for _, row in track.iterrows():
            if prev_row is not None:
                dx = row["x_center"] - prev_row["x_center"]
                dy = row["y_center"] - prev_row["y_center"]
                displacement = np.hypot(dx, dy)

                pos = row["y_center"] if axis == "y" else row["x_center"]
                mpp = interpolate_scale(pos, calibration_points)
                speed = displacement * mpp * 30  # assuming 30 FPS

                speed_mps.append(speed)
                speed_kmph.append(speed * 3.6)
            else:
                speed_mps.append(0.0)
                speed_kmph.append(0.0)
            prev_row = row

    df["speed_mps"] = speed_mps
    df["speed_kmph"] = speed_kmph

    # Debug output: show speed for a sample track
    sample_track_id = df['track_id'].iloc[0]
    sample = df[df['track_id'] == sample_track_id]
    print(f"\n--- {video_id}, track_id={sample_track_id} ---")
    print(sample[['frame', 'x_center', 'y_center', 'speed_kmph']].head(10))

    df.to_csv(OUTPUT_DIR / f"{video_id}_speed_estimated.csv", index=False)
    print(f"Saved {video_id}_speed_estimated.csv")

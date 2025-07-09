import pandas as pd
import numpy as np
from pathlib import Path
import math
import csv

# === Configuration ===
POINTS_DIR = Path("behavior/GCP_arc_points")
OUTPUT_CSV = POINTS_DIR.parent / "arc_point_summary.csv"
REAL_DISTANCE_M = 15
NUM_INTERVALS = 14

# === Helper functions ===
def compute_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(-dy, dx)  # y axis goes downward in image
    angle_deg = math.degrees(angle_rad) % 360
    return round(angle_deg, 2)

def compute_arc_length(points):
    return sum(np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points)))

# === Main process ===
summary_data = []

csv_files = sorted(POINTS_DIR.glob("*_arc_points.csv"))

for file in csv_files:
    try:
        df = pd.read_csv(file)
        if df.shape[0] < 2:
            continue

        # Flexible column names
        x_col = next(c for c in df.columns if c.lower() in ['x', 'u'])
        y_col = next(c for c in df.columns if c.lower() in ['y', 'v'])

        points = df[[x_col, y_col]].to_numpy()
        p1, p2 = points[0], points[-1]

        # Compute metrics
        angle_horizontal = compute_angle(p1, p2)
        tilt_from_vertical = abs(90 - angle_horizontal)
        arc_length_px = compute_arc_length(points)
        chord_length_px = np.linalg.norm(p2 - p1)
        correction_factor = arc_length_px / chord_length_px if chord_length_px > 0 else None
        pixel_per_meter = arc_length_px / (REAL_DISTANCE_M * NUM_INTERVALS)

        # Store result
        summary_data.append({
            "file": file.name,
            "angle_deg_horizontal": angle_horizontal,
            "angle_deg_tilt_from_vertical": round(tilt_from_vertical, 2),
            "arc_length_px": round(arc_length_px, 2),
            "chord_length_px": round(chord_length_px, 2),
            "correction_factor": round(correction_factor, 4) if correction_factor else None,
            "pixels_per_meter": round(pixel_per_meter, 3)
        })

    except Exception as e:
        print(f"Error processing {file.name}: {e}")

# === Export to CSV ===
if summary_data:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"Summary saved to: {OUTPUT_CSV}")
else:
    print("No valid arc point files processed.")

import pandas as pd
import json
from pathlib import Path

# Load all *_vertical_tiers_calibration.csv files
CALIB_DIR = Path("GCP_points_yC")
OUTPUT_JSON = CALIB_DIR / "dynamic_meters_per_pixel_axis_map.json"

calibration_files = sorted(CALIB_DIR.glob("*_vertical_tiers_calibration.csv"))
calibration_axis_map = {}

for file in calibration_files:
    try:
        df = pd.read_csv(file)
        video_id = file.stem.replace("_vertical_tiers_calibration", "")

        # Check variation in X vs Y to determine dominant perspective axis
        file_img_name = video_id + ".png"
        if "C_" in video_id:
            # Horizontal distortion
            axis = "x"
        else:
            axis = "y"

        axis_values = df["y_center"].tolist() if axis == "y" else df["pixel_distance"].tolist()
        mpp_values = df["meters_per_pixel"].tolist()
        calibration_axis_map[video_id] = [axis, list(zip(axis_values, mpp_values))]

    except Exception as e:
        print(f"Error processing {file.name}: {e}")

# Save as JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(calibration_axis_map, f, indent=2)

# Show first 2 entries for inspection
list(calibration_axis_map.items())[:2]

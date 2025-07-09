import pandas as pd
import os
import glob

# Folder containing YOLOv8 detection CSVs
yolo_folder = '.'
pattern = os.path.join(yolo_folder, '*.csv')
files = sorted(glob.glob(pattern))

print("Matched input files:")
for f in files:
    print("  -", f)

# Function to group class IDs
def map_to_group(cls):
    if cls in [0, 1, 2]: return 0  # VRU
    elif cls in [3, 4]: return 1  # Fast (car, van)
    elif cls in [5, 6]: return 2  # Slow (truck, bus)
    else: return -1  # Unknown or ignored

# Process each file
for file in files:
    try:
        df = pd.read_csv(file)

        if df.empty:
            print(f"Skipped empty file: {file}")
            continue

        # Filter to only frames 0 to 200
        df = df[df["frame"].between(0, 200)]

        # Map classes to group
        df["group_class"] = df["class"].apply(map_to_group)

        # Reorder and keep relevant columns
        df = df[["frame", "group_class", "class", "confidence", "x_center", "y_center"]]

        # Save to new file
        base = os.path.basename(file)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(yolo_folder, f"{name}_grouped_f0to200.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {os.path.abspath(out_path)}")

    except Exception as e:
        print(f"Error processing {file}: {e}")
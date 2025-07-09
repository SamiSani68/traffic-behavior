import pandas as pd
import os
import glob

# Folder with ground truth CSV files
gt_folder = 'ground_truth_combined/'
pattern = os.path.join(gt_folder, '*.csv')
files = sorted(glob.glob(pattern))

print("Matched input files:")
for f in files:
    print("  -", f)

# Define class grouping logic
def map_to_group(cls):
    if cls in [0, 1, 2]: return 0  # VRU
    elif cls in [3, 4]: return 1  # Fast
    elif cls in [5, 6]: return 2  # Slow
    else: return -1

# Process each file
for file in files:
    try:
        df = pd.read_csv(file)

        if df.empty:
            print(f"Skipped empty file: {file}")
            continue

        # Filter frames 0–200
        df = df[df["frame"].between(0, 200)]

        # Map group class
        df["group_class"] = df["class"].apply(map_to_group)

        # Decide which columns to keep
        base_columns = ["frame", "group_class", "class", "x_center", "y_center"]
        if "confidence" in df.columns:
            base_columns.insert(3, "confidence")  # Insert confidence at the right place

        df = df[[col for col in base_columns if col in df.columns]]

        # Save output
        base = os.path.basename(file)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(gt_folder, f"{name}_grouped_f0to200.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {os.path.abspath(out_path)}")

    except Exception as e:
        print(f"Error processing {file}: {e}")
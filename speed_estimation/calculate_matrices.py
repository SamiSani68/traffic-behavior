#It reads the _points.csv and _distances.csv files. It then uses OpenCV's getPerspectiveTransform function to calculate a 3x3 perspective transformation matrix (also known as a homography). This matrix can convert any pixel coordinate on the road to its real-world (x, y) coordinate in meters.
# python speed_estimation/calculate_matrices.py --input_dir gcp_data --output_dir speed_estimation/matrices
#input:speed_estimation/gcp_data
#output:speed_estimation/matrices
import cv2
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def calculate_and_save_matrices(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    points_files = list(input_dir.glob('*_points.csv'))

    if not points_files:
        print(f"Error: No '*_points.csv' files found in {input_dir}")
        return

    print(f"Found {len(points_files)} point files to process.")
    count = 0

    for points_file in points_files:
        base_name = points_file.stem.replace('_points', '')

        distances_file = input_dir / f"{base_name}_distances.csv"

        if not distances_file.exists():
            print(f"Warning: Found {points_file.name}, but missing {distances_file.name}. Skipping.")
            continue

        print(f"\n--- Processing: {base_name} ---")

        points_df = pd.read_csv(points_file)
        distances_df = pd.read_csv(distances_file)

        source_points = np.array(points_df.values, dtype="float32")

        # Destination points are the ideal, top-down rectangle.
        # We use the real-world distances to define its dimensions.
        # Based on our prompt order:
        # dist 1->2 is the width
        # dist 2->3 is the length
        width = distances_df['dist_m'][0]  # First row (1 -> 2)
        length = distances_df['dist_m'][1]  # Second row (2 -> 3)

        destination_points = np.array([
            [0, 0],  # Point 1 (Top-Left)
            [width, 0],  # Point 2 (Top-Right)
            [width, length],  # Point 3 (Bottom-Right)
            [0, length]  # Point 4 (Bottom-Left)
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(source_points, destination_points)

        output_path = output_dir / f"{base_name}_matrix.npy"
        np.save(output_path, matrix)

        print(f"Successfully calculated and saved matrix to {output_path}")
        count += 1

    print(f"\nFinished. Processed {count} pairs of files.")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and save perspective transformation matrices from GCP data.")
    parser.add_argument("--input_dir", required=True, type=Path,
                        help="Directory containing the '_points.csv' and '_distances.csv' files.")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Directory to save the output '_matrix.npy' files.")
    args = parser.parse_args()

    calculate_and_save_matrices(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
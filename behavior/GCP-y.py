import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import csv
import sys
import cv2
import matplotlib
matplotlib.use('TkAgg')

# === Config ===
IMAGE_DIR = Path("behavior/GCP")
CSV_OUT = Path("behavior/GCP_points_csv_multi")
H_OUT = Path("behavior/homography_matrices_multi")
CSV_OUT.mkdir(parents=True, exist_ok=True)
H_OUT.mkdir(parents=True, exist_ok=True)

REGIONS = ["bottom", "middle", "top"]
POINTS_PER_REGION = 4
TOTAL_POINTS = len(REGIONS) * POINTS_PER_REGION

real_world_quad = np.array([
    [0, 0],
    [30, 0],
    [30, 25],
    [0, 25]
], dtype=np.float32)

current_pixel_points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        u, v = int(event.xdata), int(event.ydata)
        current_pixel_points.append((u, v))
        ax.plot(u, v, 'ro', markersize=6)
        fig.canvas.draw_idle()
        print(f"Clicked: ({u}, {v})")
        if len(current_pixel_points) >= TOTAL_POINTS:
            print(f"Collected {TOTAL_POINTS} points. Closing window.")
            plt.close()

def on_key(event):
    if hasattr(event, "key") and event.key == 'q':
        print("Skipped image manually with 'q'.")
        current_pixel_points.clear()
        plt.close()

# === Image loop ===
print(f"Looking for .png images in: {IMAGE_DIR}")
if not IMAGE_DIR.exists():
    print(f"Error: folder not found: {IMAGE_DIR}")
    sys.exit(1)

for image_path in sorted(IMAGE_DIR.glob("*.png")):
    current_pixel_points = []
    print(f"Processing: {image_path.name}")
    print(f"Click 4 points for each region: {', '.join(REGIONS)}. Total = {TOTAL_POINTS}. Press 'q' to skip.")

    try:
        img = Image.open(image_path)
        img_np = np.array(img)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_np)
        ax.set_title(f"Click 4 points for each of: {', '.join(REGIONS)}")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

        if len(current_pixel_points) == TOTAL_POINTS:
            for i, region in enumerate(REGIONS):
                pts = current_pixel_points[i * POINTS_PER_REGION:(i + 1) * POINTS_PER_REGION]
                csv_path = CSV_OUT / f"{image_path.stem}_{region}_points.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["u", "v"])
                    writer.writerows(pts)
                print(f"Saved CSV: {csv_path.name}")

                pixel_coords = np.array(pts, dtype=np.float32)
                H, _ = cv2.findHomography(pixel_coords, real_world_quad)
                np.save(H_OUT / f"{image_path.stem}_{region}_H.npy", H)
                print(f"Saved H matrix: {image_path.stem}_{region}_H.npy")
        else:
            print(f"Not enough points collected ({len(current_pixel_points)}/{TOTAL_POINTS}). Skipping.")

    except Exception as e:
        print(f"Error with {image_path.name}: {e}")

print("Done with all images.")
print(f"CSVs saved in: {CSV_OUT}")
print(f"Homographies saved in: {H_OUT}")

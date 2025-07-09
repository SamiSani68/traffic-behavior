import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import csv
import sys

# === Configuration ===
IMAGE_DIR = Path("behavior/GCP")
OUTPUT_DIR = Path("behavior/GCP_points_csv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_POINTS = 4  # For homography, use exactly 4 points
current_pixel_points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        u, v = int(event.xdata), int(event.ydata)
        current_pixel_points.append((u, v))
        ax.plot(u, v, 'ro', markersize=6)
        fig.canvas.draw_idle()
        print(f"Clicked: ({u}, {v})")

        if len(current_pixel_points) >= MAX_POINTS:
            print(f"\n✅ Collected {MAX_POINTS} points. Closing window.")
            plt.close()

def on_key(event):
    if event.key == 'q':
        print("Skipped image manually with 'q'.")
        current_pixel_points.clear()
        plt.close()

print(f"Looking for .png images in: {IMAGE_DIR}")

if not IMAGE_DIR.exists():
    print(f"Error: Image directory not found: {IMAGE_DIR}")
    sys.exit(1)

for image_path in sorted(IMAGE_DIR.glob("*.png")):
    current_pixel_points = []
    print(f"Processing: {image_path.name}")
    print("Click 4 points on the road plane (see order in instructions). Press 'q' to skip.")

    try:
        img = Image.open(image_path)
        img_np = np.array(img)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_np)
        ax.set_title(f"Click 4 GCP points on: {image_path.name}")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

        # Save if 4 points were collected
        if len(current_pixel_points) == MAX_POINTS:
            csv_path = OUTPUT_DIR / f"{image_path.stem}_pixel_points.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["u", "v"])
                writer.writerows(current_pixel_points)
            print(f"Saved: {csv_path}")
        else:
            print(f"Not enough points collected for {image_path.name}. Skipped saving.")

    except Exception as e:
        print(f"❌ Error with {image_path.name}: {e}")

print("GCP collection finished.")
print(f"Saved CSVs in: {OUTPUT_DIR}")

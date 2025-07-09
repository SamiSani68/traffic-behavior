import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import csv
import sys
import math

# === Configuration ===
IMAGE_DIR = Path("behavior/GCP")
POINTS_CSV_DIR = Path("behavior/GCP_arc_points")
POINTS_CSV_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV_PATH = IMAGE_DIR / "arc_point_angles.csv"
MAX_POINTS = 15

current_arc_points = []
summary_rows = []

def compute_path_angle(p1, p2):
    """Compute angle in degrees between horizontal axis and vector p1 → p2"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(-dy, dx)  # negative dy: image y-axis goes down
    angle_deg = math.degrees(angle_rad)
    return round(angle_deg % 360, 2)

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        u, v = int(event.xdata), int(event.ydata)
        current_arc_points.append((u, v))
        ax.plot(u, v, 'ro', markersize=5)
        fig.canvas.draw_idle()
        print(f"🖱️ Clicked: ({u}, {v})")

        if len(current_arc_points) >= MAX_POINTS:
            print(f"\n✅ Collected {MAX_POINTS} points. Closing window.")
            plt.close()

def on_key(event):
    if event.key == 'q':
        print("Skipped image manually with 'q'.")
        current_arc_points.clear()
        plt.close()

print(f"Looking for .png images in: {IMAGE_DIR}")
if not IMAGE_DIR.exists():
    print(f"Error: Image directory not found: {IMAGE_DIR}")
    sys.exit(1)

for image_path in sorted(IMAGE_DIR.glob("*.png")):
    current_arc_points = []
    print(f"Processing: {image_path.name}")
    print("Click 15 points along the road (each ~15m apart). Press 'q' to skip.")

    try:
        img = Image.open(image_path)
        img_np = np.array(img)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_np)
        ax.set_title(f"{image_path.name} — Click 15 points on road (Press 'q' to skip)")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

        if len(current_arc_points) == MAX_POINTS:
            # Save arc points CSV
            csv_path = POINTS_CSV_DIR / f"{image_path.stem}_arc_points.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                writer.writerows(current_arc_points)
            print(f"Saved: {csv_path}")

            # Compute angle between first and last point
            angle_deg = compute_path_angle(current_arc_points[0], current_arc_points[-1])
            summary_rows.append([image_path.name, angle_deg])
            print(f"Angle with horizontal: {angle_deg:.2f}°")

        else:
            print(f"Not enough points collected for {image_path.name}. Skipping.")

    except Exception as e:
        print(f"Error with {image_path.name}: {e}")

# === Save final summary CSV ===
if summary_rows:
    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "angle_deg"])
        writer.writerows(summary_rows)
    print(f"Saved summary angles to: {SUMMARY_CSV_PATH}")
else:
    print("No summary saved. No valid images processed.")

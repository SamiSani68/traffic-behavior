import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure pixel distance between two user‑clicked segments in an image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("behavior/GCP/A_50m.png"),
        help="Path to the frame image [default: behavior/GCP/A_50m.png]",
    )
    parser.add_argument(
        "--vertical_m",
        type=float,
        default=None,
        help="Real‑world distance (metres) for the first vertical segment",
    )
    parser.add_argument(
        "--horizontal_m",
        type=float,
        default=None,
        help="Real‑world distance (metres) for the second horizontal segment",
    )
    return parser.parse_args()


args = parse_args()

if not args.image.exists():
    sys.exit(f"Image not found: {args.image}\nPass --image path or place the default frame there.")

POINTS_NEEDED = 4
clicked_pts: list[tuple[float, float]] = []  # (x, y)


# ───────────────────────────────────────────────────────────────────────────────
# Callback for Matplotlib clicks
# ───────────────────────────────────────────────────────────────────────────────

def on_click(event):
    # ignore clicks outside axes
    if event.xdata is None or event.ydata is None:
        return

    clicked_pts.append((event.xdata, event.ydata))
    idx = len(clicked_pts)
    ax.plot(event.xdata, event.ydata, "ro")

    if idx == 1:
        ax.set_title("Click the 2nd point of the vertical segment")
    elif idx == 2:
        ax.plot([clicked_pts[0][0], clicked_pts[1][0]],
                [clicked_pts[0][1], clicked_pts[1][1]], "r--")
        ax.set_title("Click the 1st point of the horizontal segment")
    elif idx == 3:
        ax.set_title("Click the 2nd point of the horizontal segment")
    elif idx == 4:
        ax.plot([clicked_pts[2][0], clicked_pts[3][0]],
                [clicked_pts[2][1], clicked_pts[3][1]], "b--")
        fig.canvas.draw_idle()
        plt.close()

    fig.canvas.draw_idle()


# ───────────────────────────────────────────────────────────────────────────────
# Load image & start GUI
# ───────────────────────────────────────────────────────────────────────────────

img = np.array(Image.open(args.image))

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img)
ax.set_title("Click the 1st point of the vertical segment")
fig.canvas.mpl_connect("button_press_event", on_click)
print("Click four points in the order: vertical‑A, vertical‑B, horizontal‑A, horizontal‑B.\nClose the window when done.")
plt.show()

if len(clicked_pts) < POINTS_NEEDED:
    print("Not enough points clicked. Exiting.")
    sys.exit()

# convert to numpy arrays
a = np.array(clicked_pts)
vert_px = np.linalg.norm(a[0] - a[1])
horz_px = np.linalg.norm(a[2] - a[3])

print("\n───────── Results ─────────")
print(f"Vertical segment:   {vert_px:.2f} px")
print(f"Horizontal segment: {horz_px:.2f} px")

if args.vertical_m:
    m_per_px_vert = args.vertical_m / vert_px
    print(
        f"Scale vertical:     1 px = {m_per_px_vert:.4f} m  ➜  {1 / m_per_px_vert:.1f} px per metre"
    )
if args.horizontal_m:
    m_per_px_horz = args.horizontal_m / horz_px
    print(
        f"Scale horizontal:   1 px = {m_per_px_horz:.4f} m  ➜  {1 / m_per_px_horz:.1f} px per metre"
    )

if args.vertical_m and args.horizontal_m:
    ratio = (args.vertical_m / vert_px) / (args.horizontal_m / horz_px)
    print(
        f"Anisotropy check (vertical‑scale / horizontal‑scale): {ratio:.4f} (1.0 = perfect square pixels)"
    )

print("Done.")

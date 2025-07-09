import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

# Load the image
image_path = Path("GCP/A_30m.png")
try:
    img = Image.open(image_path)
    img_np = np.array(img)
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}. Please ensure it's uploaded and accessible.")
    # Create a placeholder image if the actual image isn't found
    img_np = np.zeros((720, 1280, 3), dtype=np.uint8)
    img_np[:,:,0] = 100 # Red channel
    img_np[:,:,1] = 100 # Green channel
    img_np[:,:,2] = 255 # Blue channel (light purple)
    plt.text(100, 300, "Image Not Found. Placeholder Displayed.", color='white', fontsize=20)


fig, ax = plt.subplots(figsize=(15, 10)) # Adjust figure size for better viewing
ax.imshow(img_np)
ax.set_title("Ideal GCP Selection for Homography (Illustrative)")

# --- 1. Define a Real-World Coordinate System (Illustrative) ---
# Let's place our origin (0,0) at the bottom-right corner of the first full white dashed line.
# X-axis will run across the lanes.
# Y-axis will run along the lanes (up the road).

# Example points based on the image (these are APPROXIMATE pixel values for illustration)
# You would get these by clicking with your GCP collection script.

# --- Example Pixel Points (APPROXIMATE, you'll collect exact values) ---
# Let's say we target two parallel lines:
# Line 1 (Rightmost Lane Marker / Lane 1 boundary): This will be our X=0.0m reference
# Line 2 (Next Lane Marker to the left / Lane 1-2 boundary): This will be our X=3.5m reference (assuming 3.5m lane width)

# Points along Line 1 (X=0.0m in real-world)
p1_img = (990, 1630)  # P1: Closer point on Line 1
p2_img = (920, 1400)  # P2: 10m up from P1 on Line 1
p3_img = (850, 1150)  # P3: 20m up from P1 on Line 1
p4_img = (790, 900)   # P4: 30m up from P1 on Line 1

# Points along Line 2 (X=3.5m in real-world) - corresponds to same Y levels as Line 1 points
p5_img = (850, 1600)  # P5: Closer point on Line 2 (same Y-level as P1)
p6_img = (770, 1370)  # P6: 10m up from P5 on Line 2 (same Y-level as P2)
p7_img = (700, 1120)  # P7: 20m up from P5 on Line 2 (same Y-level as P3)
p8_img = (640, 870)   # P8: 30m up from P5 on Line 2 (same Y-level as P4)


# --- Corresponding Real-World Points (meters) ---
# X-axis: across the road (0m for right line, 3.5m for next line)
# Y-axis: along the road (0m, 10m, 20m, 30m from the bottom-most reference)

# Real-world points for Line 1 (X=0.0m)
p1_world = (0.0, 0.0) # P1: Our origin
p2_world = (0.0, 10.0) # P2: 10 meters up (along Y-axis)
p3_world = (0.0, 20.0) # P3: 20 meters up
p4_world = (0.0, 30.0) # P4: 30 meters up

# Real-world points for Line 2 (X=3.5m)
p5_world = (3.5, 0.0) # P5: 3.5 meters across (same Y as P1)
p6_world = (3.5, 10.0) # P6: 3.5 meters across, 10m up (same Y as P2)
p7_world = (3.5, 20.0) # P7: 3.5 meters across, 20m up (same Y as P3)
p8_world = (3.5, 30.0) # P8: 3.5 meters across, 30m up (same Y as P4)


# --- Plotting the points and grid ---
# Combine points for plotting
image_points_list = [p1_img, p2_img, p3_img, p4_img, p5_img, p6_img, p7_img, p8_img]
world_points_list = [p1_world, p2_world, p3_world, p4_world, p5_world, p6_world, p7_world, p8_world]

# Plot image points with labels
for i, (u, v) in enumerate(image_points_list):
    ax.plot(u, v, 'o', markersize=8, color='cyan', markeredgecolor='black', markeredgewidth=1) # Cyan for pixel points
    ax.text(u + 15, v - 15, f'P{i+1}\n({world_points_list[i][0]:.1f}m, {world_points_list[i][1]:.1f}m)',
            color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

# Draw illustrative real-world grid lines (in image coordinates)
# This is conceptual; the actual transformation is done by homography
# Draw lines connecting corresponding X points at different Y levels
ax.plot([p1_img[0], p5_img[0]], [p1_img[1], p5_img[1]], 'w--', alpha=0.7) # X=0 to X=3.5 at Y=0
ax.plot([p2_img[0], p6_img[0]], [p2_img[1], p6_img[1]], 'w--', alpha=0.7) # X=0 to X=3.5 at Y=10
ax.plot([p3_img[0], p7_img[0]], [p3_img[1], p7_img[1]], 'w--', alpha=0.7) # X=0 to X=3.5 at Y=20
ax.plot([p4_img[0], p8_img[0]], [p4_img[1], p8_img[1]], 'w--', alpha=0.7) # X=0 to X=3.5 at Y=30

# Draw lines connecting corresponding Y points along parallel lines
ax.plot([p1_img[0], p2_img[0], p3_img[0], p4_img[0]],
        [p1_img[1], p2_img[1], p3_img[1], p4_img[1]], 'w-', alpha=0.7) # Along X=0.0m line
ax.plot([p5_img[0], p6_img[0], p7_img[0], p8_img[0]],
        [p5_img[1], p6_img[1], p7_img[1], p8_img[1]], 'w-', alpha=0.7) # Along X=3.5m line


# Add a legend or key for understanding
plt.text(0.02, 0.98, 'Legend:\n🟢 = Ideal Click Point Location\n(Real-world meters)',
         transform=ax.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="k", lw=1, alpha=0.8))

# Draw an arrow for the real-world Y-axis (direction of travel)
ax.annotate('Y-axis (along road)', xy=(p4_img[0], p4_img[1]), xytext=(p4_img[0]+50, p4_img[1]-100),
            arrowprops=dict(facecolor='lime', shrink=0.05, width=2, headwidth=8),
            fontsize=10, color='lime', horizontalalignment='left')

# Draw an arrow for the real-world X-axis (across road)
ax.annotate('X-axis (across road)', xy=(p5_img[0], p5_img[1]), xytext=(p5_img[0]+150, p5_img[1]+50),
            arrowprops=dict(facecolor='lime', shrink=0.05, width=2, headwidth=8),
            fontsize=10, color='lime', verticalalignment='top')


plt.tight_layout()
plt.show()

#It uses matplotlib to draw a diagram that visually explains the concept of perspective transformation, showing how the skewed drone view is mapped to an accurate top-down view.
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_slide6_graphic():
    """
    Generates and saves the visual diagram for the perspective transformation slide.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # --- 1. Source (Left Side) ---
    ax.text(0.15, 0.9, "Source Points (Drone's View)", ha='center', fontsize=14, weight='bold')
    source_poly = patches.Polygon(
        [[0.05, 0.3], [0.25, 0.3], [0.30, 0.7], [0.0, 0.7]],
        closed=True, fill=True, facecolor='#d3d3d3', edgecolor='black', linewidth=1.5
    )
    ax.add_patch(source_poly)
    ax.text(0.02, 0.68, '1', ha='right', va='bottom', fontsize=12, color='red')
    ax.text(0.28, 0.68, '2', ha='left', va='bottom', fontsize=12, color='red')
    ax.text(0.23, 0.32, '3', ha='right', va='bottom', fontsize=12, color='red')
    ax.text(0.07, 0.32, '4', ha='left', va='bottom', fontsize=12, color='red')
    ax.text(0.15, 0.2, "(Pixel Coordinates)", ha='center', fontsize=12, style='italic')

    # --- 2. Transformation (Middle) ---
    ax.arrow(0.35, 0.5, 0.2, 0, head_width=0.03, head_length=0.02, fc='black', ec='black')
    ax.text(0.45, 0.6, "cv2.getPerspectiveTransform()", ha='center', fontsize=12)

    # Matrix representation
    matrix_text = "[[ a, b, c ],\n [ d, e, f ],\n [ g, h, 1 ]]"
    ax.text(0.45, 0.4, matrix_text, ha='center', va='top', fontsize=12,
            fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0', ec='black'))
    ax.text(0.45, 0.2, "3x3 Homography Matrix", ha='center', fontsize=12, style='italic')

    # --- 3. Destination (Right Side) ---
    ax.text(0.8, 0.9, "Destination Points (Top-Down View)", ha='center', fontsize=14, weight='bold')
    dest_poly = patches.Rectangle(
        (0.65, 0.25), 0.3, 0.5,
        fill=True, facecolor='#d3d3d3', edgecolor='black', linewidth=1.5
    )
    ax.add_patch(dest_poly)
    ax.text(0.67, 0.73, '1', ha='right', va='bottom', fontsize=12, color='red')
    ax.text(0.93, 0.73, '2', ha='right', va='bottom', fontsize=12, color='red')
    ax.text(0.93, 0.27, '3', ha='right', va='bottom', fontsize=12, color='red')
    ax.text(0.67, 0.27, '4', ha='right', va='bottom', fontsize=12, color='red')
    ax.text(0.8, 0.1, "(Metric Coordinates)", ha='center', fontsize=12, style='italic')
    ax.text(0.8, 0.78, "Width (meters)", ha='center', va='center', fontsize=11)
    ax.text(0.98, 0.5, "Length\n(meters)", ha='center', va='center', fontsize=11, rotation=-90)

    # --- Final Touches ---
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    output_filename = 'slide6_graphic.png'
    plt.savefig(output_filename, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Successfully generated and saved the graphic as: {output_filename}")


if __name__ == "__main__":
    # You might need to install matplotlib: pip install matplotlib
    create_slide6_graphic()
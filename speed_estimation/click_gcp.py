# It displays the first frame of a video and allows you to click on four points (Ground Control Points or GCPs) that form a rectangle in the real world. Afterward, it prompts you to enter the real-world distances (in meters) between these points.
# python speed_estimation/click_gcp.py --directory videos --output_dir gcp_data
# input:videos
# output:speed_estimation/gcp_data
# Google Maps link: https://www.google.com/maps/place/45°03'38.6%22N+7°32'35.8%22E/@45.0607297,7.4674518,22112m/data=!3m1!1e3!4m4!3m3!8m2!3d45.060728!4d7.543267?entry=ttu&g_ep=EgoyMDI1MDgyNS4wIKXMDSoASAFQAw%3D%3D
import cv2
import argparse
from pathlib import Path
import pandas as pd

points = []
frame_display = None
scale = 1.0
mouse_pos = (0, 0)


def on_mouse(event, x, y, flags, param):
    global frame_display, points, scale, mouse_pos

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            points.append((x_orig, y_orig))

            cv2.circle(frame_display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame_display, str(len(points)), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(points) > 1:
                prev_x_orig, prev_y_orig = points[-2]
                prev_x_display = int(prev_x_orig * scale)
                prev_y_display = int(prev_y_orig * scale)
                cv2.line(frame_display, (prev_x_display, prev_y_display), (x, y), (0, 255, 0), 2)
            if len(points) == 4:
                first_x_orig, first_y_orig = points[0]
                first_x_display = int(first_x_orig * scale)
                first_y_display = int(first_y_orig * scale)
                cv2.line(frame_display, (x, y), (first_x_display, first_y_display), (0, 255, 0), 2)
        else:
            print("4 points already selected. Press 'r' to reset or 'q' for next step.")


def prompt_for_distances():

    print("\n--- Enter Real-World Distances (in meters) ---")
    distances = []
    labels = ["Point 1 -> 2 (top width)", "Point 2 -> 3 (right length)", "Point 3 -> 4 (bottom width)",
              "Point 4 -> 1 (left length)"]
    for label in labels:
        while True:
            try:
                dist_str = input(f"Enter distance for {label}: ").strip()
                dist_float = float(dist_str)
                if dist_float > 0:
                    distances.append(dist_float)
                    break
                else:
                    print("Distance must be a positive number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    return distances


def save_data_to_csv(output_dir, video_basename, points_data, distances_data):
    points_df = pd.DataFrame(points_data, columns=['x', 'y'])
    distances_df = pd.DataFrame({
        "from": [1, 2, 3, 4],
        "to": [2, 3, 4, 1],
        "dist_m": distances_data
    })
    points_filepath = output_dir / f"{video_basename}_points.csv"
    distances_filepath = output_dir / f"{video_basename}_distances.csv"
    points_df.to_csv(points_filepath, index=False)
    distances_df.to_csv(distances_filepath, index=False)
    print(f"Successfully saved data to:\n  - {points_filepath}\n  - {distances_filepath}")


def main():
    global points, frame_display, scale, mouse_pos

    parser = argparse.ArgumentParser(description="Collect GCPs and distances and save to CSVs.")
    parser.add_argument("--directory", required=True, help="Path to the directory with .MP4 files.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to save the output CSV files.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path(args.directory)
    video_files = list(video_dir.glob('*.[mM][pP]4'))

    if not video_files:
        print(f"Error: No .MP4 files found in {args.directory}")
        return

    print(f"Found {len(video_files)} videos to process.")

    for video_path in video_files:
        points = []
        print(f"\n--- Processing: {video_path.name} ---")

        cap = cv2.VideoCapture(str(video_path))
        success, frame_original = cap.read()
        cap.release()

        if not success:
            print(f"Error: Could not read frame from {video_path.name}. Skipping.")
            continue

        h_orig, w_orig, _ = frame_original.shape
        max_display_width = 1600

        display_height = 0

        if w_orig > max_display_width:
            scale = max_display_width / w_orig
            display_height = int(h_orig * scale)
            frame_display = cv2.resize(frame_original, (max_display_width, display_height))
        else:
            scale = 1.0
            frame_display = frame_original.copy()

        window_name = f"GCP Collector: {video_path.name} | 'r' to reset, 'q' to finish"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

        print("Click on the image to select 4 points. Press 'q' when finished.")

        while True:
            canvas = frame_display.copy()
            if mouse_pos[0] > 0 and mouse_pos[1] > 0:
                cv2.line(canvas, (mouse_pos[0] - 20, mouse_pos[1]), (mouse_pos[0] + 20, mouse_pos[1]), (255, 255, 0), 1)
                cv2.line(canvas, (mouse_pos[0], mouse_pos[1] - 20), (mouse_pos[0], mouse_pos[1] + 20), (255, 255, 0), 1)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                points = []
                if w_orig > max_display_width:
                    frame_display = cv2.resize(frame_original, (max_display_width, display_height))
                else:
                    frame_display = frame_original.copy()
                print("Points reset for the current video.")

        cv2.destroyAllWindows()

        if len(points) == 4:
            distances = prompt_for_distances()
            video_basename = video_path.stem
            save_data_to_csv(args.output_dir, video_basename, points, distances)
        else:
            print(f"Warning: Only {len(points)} points collected for {video_path.name}. No data saved.")

    print("\n\nAll videos processed.")


if __name__ == "__main__":
    main()

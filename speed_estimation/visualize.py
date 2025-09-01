#It reads a _final_speeds.csv file and the original video. It then draws bounding boxes on each vehicle and annotates them with their calculated speed in km/h. It will color the box red for speeding vehicles.
# python speed_estimation/visualize.py --speeds_dir speed_estimation/final_output_smooth --videos_dir videos --output_dir speed_estimation/annotated_videos_avg

import cv2
import argparse
from pathlib import Path
import pandas as pd


def visualize_speeds(video_path: Path, tracks_df: pd.DataFrame, output_path: Path):
    print(f"  - Loading video: {video_path.name}")

    tracks_by_frame = {int(frame): data for frame, data in tracks_df.groupby('frame')}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # Get FPS as float first
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, int(fps), (width, height))

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in tracks_by_frame:
            frame_data = tracks_by_frame[frame_num]

            for _, row in frame_data.iterrows():
                x_center = row['x_center']
                y_center = row['y_center']
                w = row['width']
                h = row['height']
                speed = row['speed_kph']
                class_id = int(row['class'])

                x1 = int(x_center - (w / 2))
                y1 = int(y_center - (h / 2))
                x2 = int(x1 + w)
                y2 = int(y1 + h)

                box_color = (0, 255, 0)
                if (class_id == 1 and speed > 130) or \
                   (class_id == 2 and speed > 100):
                    box_color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"{speed:.1f}"

                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_y = max(y1, label_size[1] + 10)
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), (x1 + label_size[0], label_y - base_line),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (x1, label_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        out.write(frame)

        if frame_num % 100 == 0:
            print(f"  - Processing frame {frame_num}/{total_frames}", end='\r')

        frame_num += 1

    print(f"\n  - Finished processing. Annotated video saved to: {output_path}")
    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(description="Visualize speed tracking data on videos.")
    parser.add_argument("--speeds_dir", required=True, type=Path,
                        help="Directory with the final '_final_speeds.csv' files.")
    parser.add_argument("--videos_dir", required=True, type=Path, help="Directory with the original .MP4 video files.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to save the new annotated videos.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    speed_files = list(args.speeds_dir.glob('*_final_speeds.csv'))
    if not speed_files:
        print(f"Error: No '*_final_speeds.csv' files found in {args.speeds_dir}")
        return

    for csv_path in speed_files:
        base_name = csv_path.stem.replace('_final_speeds', '')
        print(f"\n--- Starting visualization for: {base_name} ---")

        video_path = args.videos_dir / f"{base_name}.MP4"
        if not video_path.exists():
            video_path = args.videos_dir / f"{base_name}.mp4"
            if not video_path.exists():
                print(f"  - Warning: Video file not found for {base_name}. Skipping.")
                continue

        output_path = args.output_dir / f"{base_name}_annotated.mp4"

        try:
            tracks_df = pd.read_csv(csv_path)
            tracks_df.columns = tracks_df.columns.str.strip().str.lower()
        except Exception as e:
            print(f"  - Error loading CSV file for {base_name}: {e}. Skipping.")
            continue

        visualize_speeds(video_path, tracks_df, output_path)

    print("\n\nAll videos have been processed.")


if __name__ == "__main__":
    main()


# python speed_estimation/calculate_speed_per_frame_smooth.py   --tracks_dir video-analysis/tracked_videos/deepsort   --matrices_dir speed_estimation/matrices   --videos_dir videos   --output_dir speed_estimation/final_output_per_frame_smooth

import cv2
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

MIN_SPEED_KPH = 15
MAX_SPEED_KPH = 200

def calculate_speeds(tracks_df: pd.DataFrame, matrix: np.ndarray, fps: float) -> pd.DataFrame:

    FRAME_COL = 'frame'
    TRACK_ID_COL = 'track_id'
    X1_COL, Y1_COL, X2_COL, Y2_COL = 'x1', 'y1', 'x2', 'y2'

    SMOOTHING_WINDOW_SIZE = 5

    tracks_df = tracks_df.sort_values(by=[TRACK_ID_COL, FRAME_COL])

    pixel_points = tracks_df[[X1_COL, X2_COL, Y2_COL]].copy()
    pixel_points['center_x'] = (pixel_points[X1_COL] + pixel_points[X2_COL]) / 2
    pixel_points_np = pixel_points[['center_x', Y2_COL]].values.astype(np.float32)

    real_world_points = cv2.perspectiveTransform(pixel_points_np.reshape(-1, 1, 2), matrix)
    real_world_points = real_world_points.reshape(-1, 2)

    tracks_df['real_x'] = real_world_points[:, 0]
    tracks_df['real_y'] = real_world_points[:, 1]

    tracks_df[['dx', 'dy', 'df']] = tracks_df.groupby(TRACK_ID_COL)[['real_x', 'real_y', FRAME_COL]].diff()

    distance_meters = np.sqrt(tracks_df['dx'] ** 2 + tracks_df['dy'] ** 2)
    time_seconds = tracks_df['df'] / fps

    speed_kph = (distance_meters / time_seconds) * 3.6
    tracks_df['speed_kph'] = speed_kph

    tracks_df['speed_kph'] = tracks_df.groupby(TRACK_ID_COL)['speed_kph'].bfill()

    smoothed_speeds = tracks_df.groupby(TRACK_ID_COL)['speed_kph'].rolling(window=SMOOTHING_WINDOW_SIZE,
                                                                           min_periods=1).mean()

    tracks_df['speed_kph'] = smoothed_speeds.reset_index(level=0, drop=True)

    tracks_df = tracks_df.drop(columns=['real_x', 'real_y', 'dx', 'dy', 'df'])

    return tracks_df


def main():
    parser = argparse.ArgumentParser(
        description="Estimate vehicle speeds using tracking data and perspective matrices.")
    parser.add_argument("--tracks_dir", required=True, type=Path,
                        help="Directory with the original DeepSORT tracking CSV files.")
    parser.add_argument("--matrices_dir", required=True, type=Path,
                        help="Directory with the calculated '_matrix.npy' files.")
    parser.add_argument("--videos_dir", required=True, type=Path,
                        help="Directory with the original .MP4 video files to get FPS.")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Directory to save the final CSV files with speed calculations.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    track_files = list(args.tracks_dir.glob('*.csv'))
    if not track_files:
        print(f"Error: No tracking CSV files found in {args.tracks_dir}")
        return

    for track_file in track_files:
        base_name = '_'.join(track_file.stem.split('_')[:2])

        print(f"\n--- Processing: {base_name} ---")

        matrix_file = args.matrices_dir / f"{base_name}_matrix.npy"
        video_file = args.videos_dir / f"{base_name}.MP4"

        if not matrix_file.exists():
            print(f"  - Warning: Matrix file not found at {matrix_file}. Skipping.")
            continue
        if not video_file.exists():
            video_file = args.videos_dir / f"{base_name}.mp4"
            if not video_file.exists():
                print(f"  - Warning: Video file not found for {base_name}. Skipping.")
                continue

        matrix = np.load(matrix_file)
        tracks_df = pd.read_csv(track_file)

        tracks_df.columns = tracks_df.columns.str.strip().str.lower()

        tracks_df['x1'] = tracks_df['x_center'] - (tracks_df['width'] / 2)
        tracks_df['y1'] = tracks_df['y_center'] - (tracks_df['height'] / 2)
        tracks_df['x2'] = tracks_df['x1'] + tracks_df['width']
        tracks_df['y2'] = tracks_df['y1'] + tracks_df['height']

        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps == 0:
            print(f"  - Warning: Could not get FPS for {video_file.name}. Skipping.")
            continue

        tracks_with_speeds_df = calculate_speeds(tracks_df, matrix, fps)

        original_rows = len(tracks_with_speeds_df)
        filtered_df = tracks_with_speeds_df[
            tracks_with_speeds_df['speed_kph'].between(MIN_SPEED_KPH, MAX_SPEED_KPH)].copy()
        filtered_rows = len(filtered_df)
        print(
            f"  - Filtered {original_rows - filtered_rows} rows with speeds outside the range ({MIN_SPEED_KPH}-{MAX_SPEED_KPH} km/h).")

        if filtered_df.empty:
            print("  - No data remaining after filtering. No file will be saved.")
            continue

        filtered_df.sort_values(by='track_id', inplace=True)

        output_path = args.output_dir / f"{base_name}_final_speeds.csv"
        filtered_df['speed_kph'] = filtered_df['speed_kph'].round(2)
        filtered_df.to_csv(output_path, index=False)
        print(f"  - Successfully calculated smoothed speeds and saved filtered results to {output_path}")

    print("\n\nAll videos processed.")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# === Configuration ===
FPS = 30  # Frames per second of the video
MPS_TO_KPH = 3.6  # Conversion factor

# === Directory configuration ===
PROJECTED_INPUT_DIR = Path("behavior/projected_tracks")
VIDEO_INPUT_DIR = Path("video-analysis/videos")
SPEED_OUTPUT_DIR = Path("video-analysis/tracked_videos/projected_speeds")
ANNOTATED_OUTPUT_DIR = Path("video-analysis/annotated_videos")

SPEED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
COLOR = (0, 255, 0)

def compute_speeds(df):
    df = df.sort_values(by=["track_id", "frame"]).reset_index(drop=True)
    df["speed_kph"] = np.nan

    for track_id in df["track_id"].unique():
        track_df = df[df["track_id"] == track_id]
        dx = track_df["x_ground"].diff()
        dy = track_df["y_ground"].diff()
        dt = track_df["frame"].diff() / FPS
        speed_mps = np.sqrt(dx**2 + dy**2) / dt
        df.loc[track_df.index, "speed_kph"] = speed_mps * MPS_TO_KPH

    return df

def annotate_video(video_name, speed_df):
    video_path = VIDEO_INPUT_DIR / f"{video_name}.MP4"
    output_path = ANNOTATED_OUTPUT_DIR / f"{video_name}_annotated.MP4"

    if not video_path.exists():
        print(f"Video not found: {video_path.name}")
        return

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = speed_df[speed_df["frame"] == frame_idx]
        for _, row in frame_data.iterrows():
            x, y, w, h = int(row["x_center"]), int(row["y_center"]), int(row["width"]), int(row["height"])
            speed = row["speed_kph"]
            if not np.isnan(speed):
                speed_text = f'{int(speed)} km/h'
                cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), COLOR, 2)
                cv2.putText(frame, speed_text, (x - w//2, y - h//2 - 10), FONT, FONT_SCALE, COLOR, THICKNESS)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Annotated video saved: {output_path.name}")

# === Process all projected track files ===
for file in sorted(PROJECTED_INPUT_DIR.glob("*_projected.csv")):
    print(f"Processing: {file.name}")
    df = pd.read_csv(file)
    df_out = compute_speeds(df)

    # Save computed speeds
    speed_csv_path = SPEED_OUTPUT_DIR / file.name.replace("_projected.csv", "_speeds.csv")
    df_out.to_csv(speed_csv_path, index=False)
    print(f"Saved speeds: {speed_csv_path.name}")

    # Annotate and save video
    video_name = file.stem.replace("_projected", "")
    annotate_video(video_name, df_out)

print("All videos processed and annotated.")

#Extracts image frames from raw videos.
#python segmentation/frames.py
#input:videos
#output:segmentation/dataset/frames

import cv2
from pathlib import Path

video_folder = Path("videos")
output_folder = Path("dataset/frames")
output_folder.mkdir(parents=True, exist_ok=True)

frame_interval = 30  # Save 1 frame every 30 frames (~1 per second at 30fps)

for video_path in video_folder.glob('*.MP4'):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    saved_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_path = output_folder / f"{video_path.stem}_frame{saved_idx:04d}.png"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()

print("Frames extracted!")

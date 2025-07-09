import subprocess
from pathlib import Path

input_dir = Path("predictions")  # Folder with segmented .MP4 videos
model_path = "runs/detect/yolov8-fine-tuned52/weights/best.pt"
script_path = "video-analysis/detect-on-segmented.py"
confidence = 0.3

video_files = sorted(input_dir.glob("*.mp4"))

if not video_files:
    print("No .MP4 videos found in the 'predictions/' folder.")
else:
    print(f"Found {len(video_files)} videos. Starting batch detection...\n")

for video_path in video_files:
    print(f"Processing: {video_path.name}")
    subprocess.run([
        "python", script_path,
        "--video_path", str(video_path),
        "--model_path", model_path,
        "--confidence", str(confidence)
    ])
    print(f"Done: {video_path.name}\n")

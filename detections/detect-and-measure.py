# python detections/detect-and-measure.py --input_dir "videos" --model_path "runs/detect/yolov8-fine-tuned52/weights/best.pt" --confidence 0.4
# output: video-analysis/detection_results/
import os
import cv2
import argparse
import pandas as pd
from ultralytics import YOLO
from pathlib import Path


def run_detection_on_video(video_path, model, confidence):
    print(f"\n--- Processing video: {video_path.name} ---")
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    output_video_dir = "video-analysis/detection_results/annotated_videos"
    output_log_dir = "video-analysis/detection_results/logs"
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_log_dir, exist_ok=True)

    output_video_path = os.path.join(output_video_dir, f"{video_name}_annotated.avi")
    log_file_path = os.path.join(output_log_dir, f"{video_name}_detections.csv")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_number = 0
    logs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=confidence)
        detections = results[0].boxes

        for box in detections:
            cls_id = int(box.cls)
            conf_score = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            height_box = y2 - y1
            if cls_id == 4 and height_box > 40:
                cls_id = 3

            logs.append({
                "frame": frame_number,
                "class": cls_id,
                "confidence": conf_score,
                "x_center": cx,
                "y_center": cy,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            label = f"{cls_id} ({conf_score:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x, text_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + 30
            cv2.rectangle(frame,
                          (text_x, text_y - text_size[1] - 6),
                          (text_x + text_size[0] + 4, text_y + 4),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, (text_x + 2, text_y),
                        font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

    pd.DataFrame(logs).to_csv(log_file_path, index=False)
    print(f"  -> Video saved to {output_video_path}")
    print(f"  -> Detections logged to {log_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect vehicles in a directory of drone videos using YOLOv8")

    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing input videos")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained YOLOv8 model")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    args = parser.parse_args()

    print(f"Loading YOLO model from: {args.model_path}")
    model = YOLO(args.model_path)
    print("Model loaded successfully.")

    video_dir = Path(args.input_dir)
    video_files = sorted(list(video_dir.glob("*.mp4"))) + sorted(list(video_dir.glob("*.MP4")))

    if not video_files:
        print(f"Error: No .mp4 or .MP4 videos found in '{args.input_dir}'")
    else:
        print(f"Found {len(video_files)} videos to process. Starting batch detection...")

    for video_path in video_files:
        run_detection_on_video(
            video_path=video_path,
            model=model,
            confidence=args.confidence
        )

    print("\nBatch processing complete.")

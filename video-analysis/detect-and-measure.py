# detect_and_measure.py
import os
import cv2
import argparse
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

def run_detection_on_video(video_path, model_path, confidence):
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Prepare output directories
    output_video_dir = "video-analysis/results/annotated_videos"
    output_log_dir = "video-analysis/results/logs"
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_log_dir, exist_ok=True)

    # Output paths
    output_video_path = os.path.join(output_video_dir, f"{video_name}_annotated.avi")
    log_file_path = os.path.join(output_log_dir, f"{video_name}_detections.csv")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    model = YOLO(model_path)
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
                "y_center": cy
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

    # Save log CSV
    pd.DataFrame(logs).to_csv(log_file_path, index=False)
    print(f"Video saved to {output_video_path}")
    print(f"Detections logged to {log_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect vehicles in drone videos using YOLOv8")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained YOLOv8 model")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    args = parser.parse_args()

    run_detection_on_video(
        video_path=args.video_path,
        model_path=args.model_path,
        confidence=args.confidence
    )

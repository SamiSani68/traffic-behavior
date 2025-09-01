#python detections/detect-on-segmented.py --input_dir "segmentation/predictions/" --model_path "runs/detect/yolov8-fine-tuned52/weights/best.pt" --confidence 0.4
#output: video-analysis/road-only-detection/
import os
import cv2
import argparse
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

def extract_foreground_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = (35, 40, 40)
    upper_green = (85, 255, 255)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask

def run_yolo_on_segmented_video(video_path, model_path, confidence):
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    output_dir = "video-analysis/road-only-detection"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{video_name}_yolo_foreground.avi")
    log_file_path = os.path.join(output_dir, f"{video_name}_detections.csv")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    model = YOLO(model_path)
    frame_number = 0
    logs = []

    group_names = ["VRU", "Fast", "Slow"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply green-mask-based foreground filtering
        mask = extract_foreground_mask(frame)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        results = model(masked_frame, verbose=False, conf=confidence)
        detections = results[0].boxes

        for box in detections:
            raw_cls = int(box.cls)
            conf_score = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Class mapping to 3 groups
            if raw_cls in [0, 1, 2]:      # person, bicycle, motorcycle
                cls_id = 0  # VRU
            elif raw_cls in [3, 4]:       # car, van
                cls_id = 1  # Fast
            elif raw_cls in [5, 6]:       # truck, bus
                cls_id = 2  # Slow
            else:
                continue

            logs.append({
                "frame": frame_number,
                "class": cls_id,
                "label": group_names[cls_id],
                "confidence": conf_score,
                "x_center": cx,
                "y_center": cy
            })

            # Draw on original frame (not the masked one)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{group_names[cls_id]} ({conf_score:.2f})"
            cv2.putText(frame, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    pd.DataFrame(logs).to_csv(log_file_path, index=False)
    print(f"Output video saved: {output_video_path}")
    print(f"Detection log saved: {log_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLO detection on a directory of segmented videos.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the directory containing segmented videos')
    parser.add_argument('--model_path', type=str, required=True, help='Trained YOLOv8 model path')
    parser.add_argument('--confidence', type=float, default=0.3, help='YOLO detection confidence')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    video_files = sorted(list(input_dir.glob("*.mp4")))

    if not video_files:
        print(f"No .mp4 videos found in '{args.input_dir}'")
    else:
        print(f"Found {len(video_files)} videos. Starting batch processing...")

    for video_path in video_files:
        print(f"\n--- Processing: {video_path.name} ---")
        run_yolo_on_segmented_video(
            video_path=str(video_path),
            model_path=args.model_path,
            confidence=args.confidence
        )

    print("\nBatch processing complete.")
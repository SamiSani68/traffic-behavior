from ultralytics import YOLO
import os
import cv2
from pathlib import Path

def run_yolo_inference(input_video_path, output_dir, model_path="yolov8n.pt", conf=0.3):
    """
    Run YOLOv8 inference on a video and save per-frame results.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[YOLOv8] Loading model from {model_path}...")
    model = YOLO(model_path)

    print(f"[YOLOv8] Reading video from {input_video_path}...")
    cap = cv2.VideoCapture(input_video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf)
        detections = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        frame_results = []
        for box, cls_id, score in zip(detections, class_ids, scores):
            x1, y1, x2, y2 = box
            frame_results.append(f"{frame_idx},{int(cls_id)},{score:.2f},{int(x1)},{int(y1)},{int(x2)},{int(y2)}")

        # Save detections as txt
        output_file = Path(output_dir) / f"{frame_idx:06d}.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(frame_results))

        frame_idx += 1

    cap.release()
    print(f"[YOLOv8] Detection complete. Results saved to {output_dir}")

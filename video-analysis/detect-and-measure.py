import cv2
import argparse
from ultralytics import YOLO
from utils_video import estimate_distance

def detect_vehicles(
    video_path='video-analysis/videos/A_40m.mp4',
    model_path='yolov8n.pt',
    pixel_per_meter=20,
    confidence_threshold=0.3
):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection
        results = model(frame)[0]
        detections = results.boxes

        for box in detections:
            if box.conf < confidence_threshold:
                continue

            # Get box center and dimensions
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = int(box.cls[0])
            conf = float(box.conf[0])
            width_px = x2 - x1

            # Estimate real-world distance based on pixel width
            distance = estimate_distance(width_px, pixel_per_meter)

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"{model.names[label]} {conf:.2f} - {distance:.1f}m"
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Show frame
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vehicle detection on drone videos")
    parser.add_argument("--video", type=str, default="video-analysis/videos/A_40m.mp4", help="Path to the input video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--ppm", type=float, default=20.0, help="Pixels per meter")
    args = parser.parse_args()

    detect_vehicles(video_path=args.video, model_path=args.model, pixel_per_meter=args.ppm)

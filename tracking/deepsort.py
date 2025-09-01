#apply the DeepSORT algorithm, which uses both motion and visual appearance (the frame itself) to track objects.
#python tracking/deepsort1.py
#input: video-analysis/results/logs , video-analysis/videos
#output: video-analysis/tracked_videos/deepsort _deepsort_tracked.mp4 , _deepsort_tracks.csv
import cv2
import pandas as pd
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort

LOG_DIR = Path("video-analysis/results/logs")
VIDEO_DIR = Path("video-analysis/videos")
OUTPUT_DIR = Path("video-analysis/tracked_videos/deepsort")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def map_class(c):
    if c in [0, 1, 2]: return 0  # VRU
    elif c in [3, 4]: return 1  # Fast
    elif c in [5, 6]: return 2  # Slow
    return -1

group_names = ["VRU", "Fast", "Slow", "Unknown"]
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (128, 128, 128)]

deep_sort = DeepSort(max_age=30)

for detection_csv in sorted(LOG_DIR.glob("*_detections.csv")):
    base_name = detection_csv.stem.replace("_detections", "")
    video_path = VIDEO_DIR / f"{base_name}.MP4"
    output_video = OUTPUT_DIR / f"{base_name}_deepsort_tracked.mp4"
    output_csv = OUTPUT_DIR / f"{base_name}_deepsort_tracks.csv"

    if not video_path.exists():
        print(f"❌ Missing video for {base_name}: {video_path}")
        continue

    print(f"Processing {base_name}")
    df = pd.read_csv(detection_csv)
    df["group_class"] = df["class"].apply(map_class)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_id = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = df[df["frame"] == frame_id]
        input_detections = []

        for _, row in dets.iterrows():
            x_center, y_center = row["x_center"], row["y_center"]
            w_box, h_box = 50, 50
            x1 = x_center - w_box / 2
            y1 = y_center - h_box / 2
            input_detections.append(([x1, y1, w_box, h_box], row["confidence"], row["group_class"]))

        tracks = deep_sort.update_tracks(input_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()
            cls = track.det_class if hasattr(track, 'det_class') else -1
            cls = int(cls) if cls in [0, 1, 2] else -1

            x1, y1, x2, y2 = map(int, ltrb)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            label = f"{group_names[cls]}-{tid}" if 0 <= cls < 3 else f"Unknown-{tid}"
            color = colors[cls] if 0 <= cls < 3 else colors[-1]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            results.append({
                "frame": frame_id,
                "track_id": tid,
                "class": cls,
                "x_center": cx,
                "y_center": cy,
                "width": x2 - x1,
                "height": y2 - y1
            })

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved: {output_video}\n CSV: {output_csv}")

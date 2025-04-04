from detections.yolo_inference import run_yolo_inference
from tracking.bytetrack_wrapper import run_tracking
from lanes.assign_lane import assign_lane_ids
from behavior.speed_estimation import estimate_speeds
from behavior.lane_change_detector import detect_lane_changes
from behavior.behavior_predictor import predict_behaviors
from visualization.annotate_video import annotate_frames

# Paths
INPUT_VIDEO = "data/processed_frames/input_video.mp4"
DETECTIONS_DIR = "detections/results/"
TRACKING_DIR = "tracking/tracker_output/"
LANE_MAP_PATH = "data/lane_maps/lane_geometry.json"
OUTPUT_VIDEO = "output/final_annotated_video.mp4"

def main():
    print("[1] Running YOLOv8 Detection...")
    run_yolo_inference(INPUT_VIDEO, DETECTIONS_DIR)

    print("[2] Running ByteTrack Tracking...")
    run_tracking(DETECTIONS_DIR, TRACKING_DIR)

    print("[3] Assigning Lane IDs...")
    assign_lane_ids(TRACKING_DIR, LANE_MAP_PATH)

    print("[4] Estimating Speed...")
    estimate_speeds(TRACKING_DIR)

    print("[5] Detecting Lane Changes...")
    detect_lane_changes(TRACKING_DIR)

    print("[6] Predicting Behaviors...")
    predict_behaviors(TRACKING_DIR)

    print("[7] Annotating Output Video...")
    annotate_frames(INPUT_VIDEO, TRACKING_DIR, OUTPUT_VIDEO)

    print("✅ Pipeline Complete. Output saved to:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
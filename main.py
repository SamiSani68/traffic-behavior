# This is the main script to run the entire video analysis pipeline.
# It executes six key scripts in sequence:
# 1. Detect objects using a fine-tuned YOLO model.
# 2. Track the detected objects using DeepSORT.
# 3. Pause for manual camera calibration by clicking GCPs.
# 4. Calculate the perspective transformation matrix.
# 5. Calculate the smoothed average speed for each tracked vehicle.
# 6. Create a final, annotated video showing the calculated speeds.

import subprocess
from pathlib import Path


def run_command(command):
    print(f"\n{'=' * 20}\n[RUNNING]: {' '.join(command)}\n{'=' * 20}")
    try:
        subprocess.run(command, check=True)
        print(f"[SUCCESS]: Command completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR]: Command failed with exit code {e.returncode}")
        print(f"--> Command was: {' '.join(command)}")
        exit(1)
    except FileNotFoundError:
        print(f"[ERROR]: The script '{command[1]}' was not found. Please check the path.")
        exit(1)


def main():
    print("--- Starting the Full Video Analysis Pipeline ---")

    # --- 1. Configuration: Define all paths and models here ---

    # Input Directories
    VIDEOS_DIR = "videos"

    # Model Path
    YOLO_MODEL_PATH = "fine-tuning/runs/detect/yolov8-fine-tuned52/weights/best.pt"

    # Intermediate Output Directories
    DETECTION_LOGS_DIR = "video-analysis/detection_results/logs"
    GCP_DATA_DIR = "speed_estimation/gcp_data"
    MATRICES_DIR = "speed_estimation/matrices"
    TRACKING_DIR = "video-analysis/tracked_videos/deepsort"
    SPEEDS_DIR = "speed_estimation/final_output_smooth"

    # Final Output Directory
    ANNOTATED_VIDEOS_DIR = "speed_estimation/annotated_videos_avg"

    # Create directories if they don't exist to prevent errors
    for dir_path in [DETECTION_LOGS_DIR, GCP_DATA_DIR, MATRICES_DIR, TRACKING_DIR, SPEEDS_DIR, ANNOTATED_VIDEOS_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # --- 2. The Pipeline Steps ---

    # Step 1: Run YOLO Detection
    print("\n--- Step 1: Running Object Detection ---")
    cmd_detect = [
        "python", "detections/detect-and-measure.py",
        "--input_dir", VIDEOS_DIR,
        "--model_path", YOLO_MODEL_PATH,
        "--confidence", "0.4"
    ]
    run_command(cmd_detect)

    # Step 2: Run DeepSORT Tracking
    print("\n--- Step 2: Running Object Tracking ---")
    cmd_track = [
        "python", "tracking/deepsort.py"
        # This script uses hardcoded paths, so no arguments needed.
    ]
    run_command(cmd_track)

    # Step 3: Manual GCP Collection (Pause for user input)
    print("\n--- Step 3: Manual Camera Calibration ---")
    print("The pipeline will now pause for manual input.")
    print("An interactive window will open for each video.")
    print("Please click 4 points and enter the real-world distances when prompted.")

    cmd_gcp = [
        "python", "speed_estimation/click_gcp.py",
        "--directory", VIDEOS_DIR,
        "--output_dir", GCP_DATA_DIR
    ]
    run_command(cmd_gcp)

    print("\n[INFO]: Manual calibration complete. Resuming automated pipeline...")

    # Step 4: Calculate Perspective Matrices
    print("\n--- Step 4: Calculating Perspective Matrices ---")
    cmd_matrix = [
        "python", "speed_estimation/calculate_matrices.py",
        "--input_dir", GCP_DATA_DIR,
        "--output_dir", MATRICES_DIR
    ]
    run_command(cmd_matrix)

    # Step 5: Calculate Smoothed Average Speeds
    print("\n--- Step 5: Calculating Vehicle Speeds ---")
    cmd_speed = [
        "python", "speed_estimation/calculate_speed_smooth.py",
        "--tracks_dir", TRACKING_DIR,
        "--matrices_dir", MATRICES_DIR,
        "--videos_dir", VIDEOS_DIR,
        "--output_dir", SPEEDS_DIR
    ]
    run_command(cmd_speed)

    # Step 6: Visualize Speeds on Video
    print("\n--- Step 6: Creating Final Annotated Videos ---")
    cmd_visualize = [
        "python", "speed_estimation/visualize.py",
        "--speeds_dir", SPEEDS_DIR,
        "--videos_dir", VIDEOS_DIR,
        "--output_dir", ANNOTATED_VIDEOS_DIR
    ]
    run_command(cmd_visualize)

    print(f"\n{'=' * 20}\nPipeline Finished Successfully!\n{'=' * 20}")
    print(f"The final annotated videos are located in: {ANNOTATED_VIDEOS_DIR}")


if __name__ == "__main__":
    main()

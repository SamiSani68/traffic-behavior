Automated Drone-Based Traffic Analysis Pipeline

This work provides an end-to-end computer vision pipeline for analysis of traffic flow from aerial drone video. The system automatically detects and tracks cars and then calculates their true speeds in kilometers per hour (km/h). By systematically addressing the problem of perspective distortion present in aerial videography, the pipeline is able to produce accurate and useful traffic data.
The solution integrates an optimized YOLOv8 object detection model, a DeepSORT algorithm for multi-object tracking, and a perspective transformation matrix derived from a manual calibration procedure for precise speed estimation. The result is a robust tool with implications for traffic management, urban planning, and road safety studies.

Core Features
High-Precision Car Detection: The system utilizes a project-specifically fine-tuned YOLOv8 model. During this fine-tuning, the model is trained to the specific visual characteristics of the project's drone feeds and therefore offers improved detection performance over generic pre-trained models.
Continuous Object Tracking: The DeepSORT algorithm is used to provide and maintain a unique ID to each identified vehicle across consecutive frames. This module plays a fundamental role in observing the trajectory and pattern of individual vehicles over time.
Perspective Correction and Camera Calibration: A script for interactive camera calibration provides a precise camera calibration through manual specification of Ground Control Points (GCPs). This generates a homography matrix that mathematically corrects for the drone's angle of view, enabling image coordinates to be converted to metric-space coordinates.
Accurate Speed Estimation: The system calculates a smoothed average speed for each vehicle it is tracking. By the use of the perspective transform matrix, the calculation accounts for the geometric distortions, so the speed measurement remains the same regardless of where an object is in the frame.
End-to-End Automated Pipeline: The entire workflow is controlled by a master script, main.py. All the component modules are executed sequentially by the master script to enable a reproducible end-to-end analysis with a single command.
Comprehensive Visualization: The pipeline generates annotated video files as its final output. The videos have a bounding box drawn around each detected vehicle, labeled with its calculated speed. Vehicles that are above predetermined speed limits can be color-coded visually for easy identification.


Project Structure
The project is organized into a modular directory structure, where each directory contains scripts dedicated to a specific stage of the pipeline.
.
├── main.py                   # The master script to run the entire pipeline
├── detections/
│   └── detect-and-measure.py # Detects objects in all videos
├── tracking/
│   └── deepsort.py           # Tracks detected objects using DeepSORT
├── speed_estimation/
│   ├── click_gcp.py          # Interactive script for manual camera calibration
│   ├── calculate_matrices.py # Calculates the perspective transformation matrix
│   ├── calculate_speed_smooth.py # Calculates the final, smooth speeds
│   └── visualize.py          # Creates the cool annotated videos
├── fine-tuning/                # Your trained YOLOv8 model should be in here
│   └── runs/detect/yolov8-fine-tuned52/weights/best.pt
├── videos/                     # Directory for your input MP4 video files
├── requirements.txt            # All the Python stuff you need to install
└── README.md                   # This file


Install Dependencies
Install the required Python libraries using the provided requirements.txt file. This command will install all necessary packages, such as OpenCV, PyTorch, and pandas.
pip install -r requirements.txt


Execution Workflow
The main.py script automates the execution of the entire pipeline.
Step 1: Input Preparation
Videos: Place all raw .MP4 video files into the videos/ directory. The pipeline is designed to process multiple videos in a batch.
Model: Confirm that the fine-tuned YOLOv8 model (best.pt) is located at fine-tuning/runs/detect/yolov8-fine-tuned52/weights/best.pt. This path can be modified in the configuration section of the main.py script if necessary.
Step 2: Execute the Pipeline
Initiate the pipeline by running the main.py script from the terminal. The script will provide progress updates for each of the six stages.
python main.py



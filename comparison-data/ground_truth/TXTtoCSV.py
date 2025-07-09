import os
import pandas as pd

# Directory paths
input_dir = '/home/fullsuper/Sami/comparison-data/ground_truth'
output_dir = 'ground_truth_combined'
os.makedirs(output_dir, exist_ok=True)

# List of video folders (A_30m, A_70m, B_50m, B_80m, C_145m)
video_folders = ['A_30m', 'A_70m', 'B_50m', 'B_80m', 'C_145m']

for video_folder in video_folders:
    video_path = os.path.join(input_dir, video_folder)
    annotations = []

    # Process each YOLO text file in the video folder
    for filename in sorted(os.listdir(video_path)):
        if filename.endswith('.txt'):
            frame_number = int(filename.split('_')[1].split('.')[0])
            file_path = os.path.join(video_path, filename)

            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # Ensure the line has the correct format
                        cls, x_center, y_center, width, height = parts[:5]
                        annotations.append([frame_number, cls, x_center, y_center, width, height])

    # Save combined annotations for this video
    output_csv = os.path.join(output_dir, f'{video_folder}_combined.csv')
    combined_df = pd.DataFrame(annotations, columns=['frame', 'class', 'x_center', 'y_center', 'width', 'height'])
    combined_df.to_csv(output_csv, index=False)

    print(f'Combined ground truth for {video_folder} saved to {output_csv}.')

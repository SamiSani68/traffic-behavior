#!/bin/bash

INPUT_DIR="behavior/deepsort_tracks"
HOMOGRAPHY_DIR="behavior/homography_matrices_multi"
OUTPUT_DIR="behavior/projected_tracks"

# Ensure output folder exists
mkdir -p "$OUTPUT_DIR"

# Loop through all *_deepsort_tracks.csv
for file in "$INPUT_DIR"/*_deepsort_tracks.csv; do
    filename=$(basename "$file")
    video_name="${filename%%_deepsort_tracks.csv}"

    echo "🔁 Projecting: $video_name"

    # Check that all 3 homography files exist
    if [[ -f "$HOMOGRAPHY_DIR/${video_name}_top_H.npy" && \
          -f "$HOMOGRAPHY_DIR/${video_name}_middle_H.npy" && \
          -f "$HOMOGRAPHY_DIR/${video_name}_bottom_H.npy" ]]; then

        # Run projection.py (assumes script reads dirs internally)
        python projection.py

    else
        echo "⚠️ Missing homography files for $video_name — skipping."
    fi
done

echo "✅ All projection tasks complete."

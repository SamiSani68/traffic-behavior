#Converts .json annotations into mask images that the model can learn from.
#python segmentation/convert-json.py
#input:segmentation/dataset/frames/images/val
#output:segmentation/dataset/frames/masked/val
import json
import cv2
import numpy as np
from pathlib import Path

json_folder = Path('dataset/frames/images/val')
output_folder = Path('dataset/frames/masked/val')
output_folder.mkdir(parents=True, exist_ok=True)

def apply_foreground_mask(json_annotation_file):
    with open(json_annotation_file, 'r') as f:
        data = json.load(f)

    image_path = json_annotation_file.parent / data['imagePath']
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for shape in data.get('shapes', []):
        if shape.get('label') == 'foreground':
            points = np.array(shape['points'], dtype=np.float32)
            points = np.round(points).astype(np.int32)  # (N,2)

            cv2.fillPoly(mask, [points], 255)  # type: ignore

    masked_image = np.zeros_like(image)
    masked_image[mask == 255] = image[mask == 255]

    output_path = output_folder / image_path.name
    cv2.imwrite(str(output_path), masked_image)
    print(f"Saved: {output_path.name}")

if __name__ == "__main__":
    all_json_files = sorted(json_folder.glob('*.json'))
    print(f"Found {len(all_json_files)} JSON files.")

    for json_file in all_json_files:
        apply_foreground_mask(json_file)

    print("All images masked successfully!")

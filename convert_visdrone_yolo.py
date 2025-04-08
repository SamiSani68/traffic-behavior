import os
import cv2
import glob
from pathlib import Path

# Mapping from VisDrone category ID to YOLOv8 class ID
# Merging tricycles (6, 7) into bicycle (1), trailers (10) into truck (5)
class_map = {
    0: 0,  # pedestrian
    1: 0,  # people
    2: 1,  # bicycle
    6: 1,  # tricycle
    7: 1,  # tricycle
    9: 2,  # motorcycle
    3: 3,  # car
    4: 4,  # van
    5: 5,  # truck
    10: 5, # trailer
    8: 6   # bus
}

def convert_visdrone_vid_split(split):
    img_dir = f"VisDrone2019-DET/{split}/sequences"
    anno_dir = f"VisDrone2019-DET/{split}/annotations"
    out_img_dir = f"dataset/images/{split}"
    out_label_dir = f"dataset/labels/{split}"

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    total_frames = 0
    skipped_annotations = 0
    missing_images = 0

    for anno_path in sorted(glob.glob(os.path.join(anno_dir, '*.txt'))):
        sequence_id = Path(anno_path).stem
        seq_img_dir = os.path.join(img_dir, sequence_id)

        if not os.path.isdir(seq_img_dir):
            print(f"❌ Image folder missing: {seq_img_dir}")
            continue

        with open(anno_path, 'r') as f:
            lines = f.readlines()

        frame_map = {}

        for line in lines:
            fields = line.strip().split(',')
            if len(fields) < 9:
                skipped_annotations += 1
                continue

            try:
                frame_id = int(fields[0])
                x, y, w, h = map(float, fields[2:6])
                class_id = int(fields[7])
            except ValueError:
                skipped_annotations += 1
                continue

            if class_id not in class_map:
                skipped_annotations += 1
                continue

            mapped_id = class_map[class_id]
            image_path = os.path.join(seq_img_dir, f"{frame_id:07}.jpg")

            if not os.path.exists(image_path):
                missing_images += 1
                continue

            img = cv2.imread(image_path)
            if img is None:
                missing_images += 1
                continue

            h_img, w_img = img.shape[:2]
            x_center = (x + w / 2) / w_img
            y_center = (y + h / 2) / h_img
            w_norm = w / w_img
            h_norm = h / h_img

            yolo_label = f"{mapped_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            img_name = f"{sequence_id}_{frame_id:07}.jpg"
            label_name = f"{sequence_id}_{frame_id:07}.txt"
            out_img_path = os.path.join(out_img_dir, img_name)

            if label_name not in frame_map:
                frame_map[label_name] = {"labels": [], "image_path": image_path, "output_image": out_img_path}

            frame_map[label_name]["labels"].append(yolo_label)

        # Write label files and copy images
        for label_file, data in frame_map.items():
            label_path = os.path.join(out_label_dir, label_file)

            with open(label_path, 'w') as lf:
                lf.write("\n".join(data["labels"]))

            if not os.path.exists(data["output_image"]):
                img = cv2.imread(data["image_path"])
                if img is not None:
                    cv2.imwrite(data["output_image"], img)

        total_frames += len(frame_map)
        print(f"✅ {sequence_id}: {len(frame_map)} frames processed")

    print(f"\n📊 Split: {split}")
    print(f"   ✅ Total labeled frames: {total_frames}")
    print(f"   ⚠️ Skipped annotations: {skipped_annotations}")
    print(f"   ⚠️ Missing or unreadable images: {missing_images}\n")


# Run for each dataset split
convert_visdrone_vid_split("train")
convert_visdrone_vid_split("val")
convert_visdrone_vid_split("test")  # Update to 'test-dev' or 'test-challenge' if needed

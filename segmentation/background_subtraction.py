#pre-trained
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import cv2
import numpy as np
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = deeplabv3_resnet101(pretrained=True)
model = model.to(device)
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TARGET_CLASSES = [2, 3, 6, 7]  # bicycle, car, bus, truck (adjust if needed)

def segment_road_and_vehicles(frame):
    original_size = (frame.shape[1], frame.shape[0])

    input_tensor = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    output_predictions = output.argmax(0).cpu().numpy()

    mask = np.zeros_like(output_predictions, dtype=np.uint8)
    for cls in TARGET_CLASSES:
        mask[output_predictions == cls] = 1

    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    masked_frame = frame.copy()
    masked_frame[mask == 0] = (0, 0, 0)

    return masked_frame


def process_all_videos(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(input_dir.glob('*.MP4')))

    for video_path in video_files:
        print(f"Processing {video_path.name}...")

        cap = cv2.VideoCapture(str(video_path))

        # Define the output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = output_dir / f"masked_{video_path.name}"
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            masked_frame = segment_road_and_vehicles(frame)
            out.write(masked_frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx} frames...")

        cap.release()
        out.release()
        print(f"Saved masked video to {out_path}")


if __name__ == "__main__":
    input_videos_dir = "video-analysis/videos/"
    output_videos_dir = "segmentation/masked_videos/"

    process_all_videos(input_videos_dir, output_videos_dir)

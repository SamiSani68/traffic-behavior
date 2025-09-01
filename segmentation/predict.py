#Loads best_model.pth and applies a green mask to the predicted foreground in videos, saving the output in the predictions folder.
#python segmentation/predict.py
#input:segmentation/checkpoints/best_model.pth , videos
#output: segmentation/predictions
import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2  # background and foreground
MODEL_PATH = 'checkpoints/best_model.pth'
INPUT_FOLDER = 'videos'
OUTPUT_FOLDER = 'predictions'

def load_model():
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict_frame(model, frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)['out']
        preds = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    return preds

def process_video(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames, desc=f"Processing {Path(video_path).name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_mask = predict_frame(model, frame)

        color_mask = np.zeros_like(frame)
        color_mask[pred_mask == 1] = [0, 255, 0]  # green for foreground

        blended = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
        out.write(blended)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()

    print(f"Saved: {output_path}")

def main():
    model = load_model()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    video_files = sorted(Path(INPUT_FOLDER).glob('*.MP4'))

    if not video_files:
        print(f"No videos found in {INPUT_FOLDER}")
        return

    for video_file in video_files:
        output_name = f"predicted_{video_file.stem}.mp4"
        output_path = Path(OUTPUT_FOLDER) / output_name
        process_video(model, str(video_file), str(output_path))

if __name__ == "__main__":
    main()

import os
import random
import shutil
from pathlib import Path

frames_folder = Path('dataset/frames')

train_folder = Path('dataset/frames/images/train')
val_folder = Path('dataset/frames/images/val')

train_folder.mkdir(parents=True, exist_ok=True)
val_folder.mkdir(parents=True, exist_ok=True)

total_samples = 200
val_ratio = 0.2

all_frames = list(frames_folder.glob('*.png'))

selected_frames = random.sample(all_frames, total_samples)
split_idx = int(total_samples * (1 - val_ratio))
train_frames = selected_frames[:split_idx]
val_frames = selected_frames[split_idx:]

for frame_path in train_frames:
    shutil.copy(frame_path, train_folder / frame_path.name)

for frame_path in val_frames:
    shutil.copy(frame_path, val_folder / frame_path.name)

print(f"Copied {len(train_frames)} frames to {train_folder}")
print(f"Copied {len(val_frames)} frames to {val_folder}")

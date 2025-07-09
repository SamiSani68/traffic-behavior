import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

DATASET_DIR = Path('dataset/frames')
IMAGE_DIR_TRAIN = DATASET_DIR / 'images' / 'train'
MASK_DIR_TRAIN = DATASET_DIR / 'masked' / 'train'
IMAGE_DIR_VAL = DATASET_DIR / 'images' / 'val'
MASK_DIR_VAL = DATASET_DIR / 'masked' / 'val'

CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

# DATASET
class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.images = sorted([img.name for img in self.image_dir.glob('*.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.images[idx]
        mask_path = self.mask_dir / self.images[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()  # Foreground = 1, Background = 0

        return image, mask

basic_transforms = transforms.Compose([
    transforms.ToTensor(),
])

def main():
    # Load datasets
    train_dataset = RoadSegmentationDataset(IMAGE_DIR_TRAIN, MASK_DIR_TRAIN, transform=basic_transforms)
    val_dataset = RoadSegmentationDataset(IMAGE_DIR_VAL, MASK_DIR_VAL, transform=basic_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Load model
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.to(DEVICE)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{NUM_EPOCHS}]')

        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.squeeze(1).long().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.squeeze(1).long().to(DEVICE)

                outputs = model(images)['out']
                loss = criterion(outputs, masks)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] -- Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / 'best_model.pth')
            print(f"✅ Best model saved at epoch {epoch+1}.")

    print(" Training completed successfully!")

if __name__ == '__main__':
    main()

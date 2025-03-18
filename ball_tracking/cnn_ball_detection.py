import os
import torch
import pandas as pd
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

class GolfBallDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        # Load bounding box (x_min, y_min, x_max, y_max)
        bbox = self.annotations.iloc[idx, 1:].values.astype(float)

        # Get original image size before resizing
        original_w, original_h = image.size  
        image = image.resize((416, 416))

        # Scale bounding box to match resized image
        x_min, y_min, x_max, y_max = bbox
        x_min = (x_min / original_w) * 416
        y_min = (y_min / original_h) * 416
        x_max = (x_max / original_w) * 416
        y_max = (y_max / original_h) * 416

        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, bbox

transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = GolfBallDataset(img_dir="train/images", labels_file="golfball_labels.csv", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)


class GolfBallCNN(nn.Module):
    def __init__(self):
        super(GolfBallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 52 * 52, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Attempted CNN improvements --> didn't actually make it better
# class GolfBallCNN(nn.Module):
#     def __init__(self):
#         super(GolfBallCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)  
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.pool = nn.MaxPool2d(2, 2)

#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc1 = nn.Linear(128, 512)
#         self.fc2 = nn.Linear(512, 4)  # Predict (x_min, y_min, x_max, y_max)

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.global_avg_pool(x)  # Convert to (batch_size, 128, 1, 1)
#         x = torch.flatten(x, start_dim=1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


def train_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GolfBallCNN().to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 30
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} starting")
        epoch_loss = 0

        for images, bboxes in dataloader:
            images, bboxes = images.to(device), bboxes.to(device)

            optimizer.zero_grad()
            with autocast(device_type="mps", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, bboxes)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_golfball_cnn.pth")
            print(f"Best model saved at epoch {epoch+1} with loss {best_loss:.4f}")

        scheduler.step()


if __name__ == "__main__":
    train_model()
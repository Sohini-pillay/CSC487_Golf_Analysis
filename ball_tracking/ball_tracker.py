import os
import cv2
import torch
import numpy as np
import pandas as pd
import random
from torchvision import transforms
from cnn_ball_detection import GolfBallCNN
import torch.nn as nn

def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

# Load model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = GolfBallCNN().to(device)
model.load_state_dict(torch.load("best_golfball_cnn.pth", map_location=device))
model.eval()

# Define loss function (Smooth L1 Loss)
criterion = nn.SmoothL1Loss()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load ground truth labels
ground_truth_df = pd.read_csv("golfball_test_labels.csv")

def predict_on_images(image_folder):
    all_images = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]
    
    iou_scores = []
    test_losses = []

    # sampled_images = random.sample(all_images, min(500, len(all_images)))

    for filename in all_images:
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load {filename}")
            continue

        h, w, _ = img.shape
        img_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = transforms.ToPILImage()(img_pil)
        image_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_bbox = model(image_tensor).cpu().numpy()[0]

        x_min_pred, y_min_pred, x_max_pred, y_max_pred = map(int, [
            predicted_bbox[0] * w / 416,
            predicted_bbox[1] * h / 416,
            predicted_bbox[2] * w / 416,
            predicted_bbox[3] * h / 416
        ])
        predicted_box = (x_min_pred, y_min_pred, x_max_pred, y_max_pred)

        gt_row = ground_truth_df[ground_truth_df["filename"] == filename]
        if gt_row.empty:
            print(f"No ground truth found for {filename}")
            continue

        x_min_gt, y_min_gt, x_max_gt, y_max_gt = gt_row.iloc[0, 1:].values
        ground_truth_box = (x_min_gt, y_min_gt, x_max_gt, y_max_gt)

        # Compute IoU
        iou = compute_iou(predicted_box, ground_truth_box)
        iou_scores.append(iou)

        # Convert ground truth and predicted bounding boxes to tensors for loss calculation
        gt_tensor = torch.tensor([x_min_gt, y_min_gt, x_max_gt, y_max_gt], dtype=torch.float32).to(device)
        pred_tensor = torch.tensor([x_min_pred, y_min_pred, x_max_pred, y_max_pred], dtype=torch.float32).to(device)

        # Compute loss for this image
        loss = criterion(pred_tensor, gt_tensor)
        test_losses.append(loss.item())

        print(f"{filename}: IoU = {iou:.4f}, Loss = {loss.item():.4f}")

        # Draw bounding boxes
        cv2.rectangle(img, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (255, 0, 0), 2)
        cv2.rectangle(img, (x_min_pred, y_min_pred), (x_max_pred, y_max_pred), (0, 255, 0), 2)

        cv2.imshow('Golf Ball Detection with IoU', img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Compute mean IoU and mean test loss
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
    mean_loss = sum(test_losses) / len(test_losses) if test_losses else 0

    print(f"\nMean IoU over {len(iou_scores)} images: {mean_iou:.4f}")
    print(f"Mean Test Loss over {len(test_losses)} images: {mean_loss:.4f}")

def main():
    image_folder = "test/images/"
    predict_on_images(image_folder)

if __name__ == "__main__":
    main()
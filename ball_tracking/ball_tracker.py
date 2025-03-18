import cv2
import torch
import numpy as np
from torchvision import transforms
from cnn_ball_detection import GolfBallCNN 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = GolfBallCNN().to(device)
model.load_state_dict(torch.load("best_golfball_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Video ended. Exiting.")
            break

        frame_count += 1

        img_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = transforms.ToPILImage()(img_pil)
        image_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_bbox = model(image_tensor).cpu().numpy()[0]

        h, w, _ = frame.shape 

        x_min, y_min, x_max, y_max = map(int, [
            predicted_bbox[0] * w / 416,
            predicted_bbox[1] * h / 416,
            predicted_bbox[2] * w / 416,
            predicted_bbox[3] * h / 416
        ])

        if x_min >= 0 and y_min >= 0 and x_max <= w and y_max <= h and (x_max - x_min) > 0 and (y_max - y_min) > 0:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "Golf Ball", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Golf Ball Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = "putt_test_2.mov"
    process_video(video_path)

if __name__ == "__main__":
    main()

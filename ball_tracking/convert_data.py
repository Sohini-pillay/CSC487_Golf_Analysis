import os
import pandas as pd
import cv2

image_dir = "train/images"
label_dir = "train/labels"

def convert_yolo_to_csv(image_dir, label_dir, output_csv):
    data = []

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            img_file = label_file.replace(".txt", ".jpg")
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, label_file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w, _ = img.shape

            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                values = line.strip().split()
                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])

                # Convert to absolute pixel coordinates
                x_min = int((x_center - width / 2) * w)
                y_min = int((y_center - height / 2) * h)
                x_max = int((x_center + width / 2) * w)
                y_max = int((y_center + height / 2) * h)

                data.append([img_file, x_min, y_min, x_max, y_max])

    # Save to CSV
    df = pd.DataFrame(data, columns=["filename", "x_min", "y_min", "x_max", "y_max"])
    df.to_csv(output_csv, index=False)

convert_yolo_to_csv(image_dir, label_dir, "golfball_labels.csv")
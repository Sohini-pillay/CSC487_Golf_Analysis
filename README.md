# **CSC 487 Final Project: Golf Swing Sequencing**

## **Project Overview**
This project focuses on analyzing and sequencing golf swing videos by detecting key frames corresponding to different stages of the swing.

## **Data Setup**
To get started, follow these steps:

1. **Download the Dataset**  
   - Download the golf swing videos from this link:  
     [Golf Swing Videos](https://drive.google.com/file/d/1uBwRxFxW04EqG87VCoX3l6vXeV5T5JYJ/view)  
   - Extract the ZIP file, which should result in a folder named **`videos_160`**.

2. **Install Dependencies**  
   - Ensure you have the required dependencies installed. You can do this using:  
     ```bash
     pip install -r requirements.txt
     ```

3. **Extract Frames**  
   - Run the following script to extract frames from the videos:  
     ```bash
     python extract_frames.py
     ```
   - This will generate a folder named **`vid_frames`**, where:
     - Each subfolder corresponds to a **video ID** from `key_frames.csv`.
     - Filenames within each subfolder include the **frame number**, which aligns with the **event index** in `key_frames.csv`.

## **Data Structure**
- **`videos_160/`**: Contains the raw golf swing videos.
- **`vid_frames/`**: Contains extracted frames from each video.
- **`key_frames.csv`**: Maps video IDs to key swing events, with frame numbers indicating specific key points in the swing sequence.
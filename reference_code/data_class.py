import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2


class GolfSwingDataset(Dataset):
    def __init__(self, frame_labels, vid_frames_dir, seq_length, transform=None, train=True):
     
        self.frame_labels = frame_labels
        self.vid_frames_dir = vid_frames_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        self.video_ids = list(frame_labels.keys())

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frame_dir = osp.join(self.vid_frames_dir, str(video_id))
        
        # Load all frame files and sort them
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png') or f.endswith('.jpg')])
        total_frames = len(frame_files)

        
        labels = [label for _, label in self.frame_labels[video_id]]

        
        images, sampled_labels = [], []
        if self.train:
        
            start_idx = np.random.randint(max(1, total_frames - self.seq_length + 1))
        else:
       
            start_idx = 0

        for i in range(start_idx, start_idx + self.seq_length):
            if i < total_frames:
                frame_path = osp.join(frame_dir, frame_files[i])
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(frame)
                sampled_labels.append(labels[i])
            else:
                # Pad with empty frames and "no-event" labels if sequence exceeds total frames
                images.append(np.zeros((160, 160, 3), dtype=np.uint8))  
                sampled_labels.append(8)  # 8 indicates "no-event"

        # Convert to tensors
        sample = {'images': np.asarray(images), 'labels': np.asarray(sampled_labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))  
        return {
            'images': torch.from_numpy(images).float().div(255.),
            'labels': torch.from_numpy(labels).long()
        }


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


if __name__ == '__main__':
 
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 

    dataset = GolfSwingDataset(
        frame_labels=frame_labels,
        vid_frames_dir='vid_frames', 
        seq_length=64,
        transform=transforms.Compose([ToTensor(), norm]),
        train=False
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0] 
        print(f'Video {i}: {len(events)} events: {events}')

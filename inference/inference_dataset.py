import torch
from torch.utils.data import Dataset
import cv2, os, numpy as np
from torchvision import transforms
from inference.init_data import faceforensics
from preprocessing.init_ff import init_ff

class ManipulationDataset(Dataset):
    def __init__(self, dataset='all', phase='test'):
        self.video_dirs, self.labels = faceforensics(dataset, phase)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                 std=[0.225, 0.225, 0.225]),
        ])

    def __len__(self):
        return len(self.video_dirs)
    
    def __getitem__(self, idx):
        folder = self.video_dirs[idx]
        label = self.labels[idx]

        frame_paths = sorted([os.path.join(folder, f)
                              for f in os.listdir(folder) if f.endswith(".png")])
        total = len(frame_paths)

        frames = []
        for i in range(total):
            img = cv2.imread(frame_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(img))

        clip = torch.stack(frames, dim=1)  # (C, T, H, W)
        return {"clip": clip, "label": torch.tensor(label, dtype=torch.long)}



class InferenceDataset(Dataset):
    def __init__(self, data_root="data", phase="test"):
        self.video_dirs, self.labels = init_ff(data_root, phase)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                 std=[0.225, 0.225, 0.225]),
        ])

    def __len__(self):
        return len(self.video_dirs)
    
    def __getitem__(self, idx):
        folder = self.video_dirs[idx]
        label = self.labels[idx]

        frame_paths = sorted([os.path.join(folder, f)
                              for f in os.listdir(folder) if f.endswith(".png")])
        total = len(frame_paths)

        frames = []
        for i in range(total):
            img = cv2.imread(frame_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(img))

        clip = torch.stack(frames, dim=1)  # (C, T, H, W)
        return {"clip": clip, "label": torch.tensor(label, dtype=torch.long)}

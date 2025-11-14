import torch
from torch.utils.data import Dataset
import cv2, os, numpy as np
from torchvision import transforms
from preprocessing.init_ff import init_ff

class FaceForensicsDataset(Dataset):
    def __init__(self, data_root="data", phase="train", num_frames=16):
        self.video_dirs, self.labels = init_ff(data_root, phase)
        self.num_frames = num_frames
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
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)

        flip = np.random.rand() < 0.5  # flip toàn clip cùng hướng
        frames = []
        for i in indices:
            img = cv2.imread(frame_paths[i])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if flip:
                img = cv2.flip(img, 1)
            frames.append(self.transform(img))

        clip = torch.stack(frames, dim=1)  # (C, T, H, W)
        return {"clip": clip, "label": torch.tensor(label, dtype=torch.long)}

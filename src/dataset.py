import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LensDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None, target_transform=None):
        self.X = np.load(images_path)
        self.y = np.load(labels_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx, :, :, :]
        label = self.y[idx]

        #convert to torch tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
            
        return image, label
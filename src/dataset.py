import os
import numpy as np
import torch
from torch.utils.data import Dataset

class QuickDrawDataset(Dataset):
    def __init__(self, data_dir, categories_file, max_per_class=5000):
        self.data = []
        self.labels = []

        with open(categories_file) as f:
            self.classes = [c.strip() for c in f.readlines()]

        for idx, cls in enumerate(self.classes):
            path = os.path.join(data_dir, f"full_numpy_bitmap_{cls}.npy")

            arr = np.load(path)[:max_per_class]

            arr = arr.reshape(-1, 28, 28).astype(np.float32)

            self.data.append(arr)
            self.labels += [idx] * len(arr)

        self.data = np.concatenate(self.data, axis=0)

        # 🚨 NORMALIZATION (единый стандарт)
        self.data = self.data / 255.0

        self.data = np.expand_dims(self.data, axis=1)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].numpy()
        label = self.labels[idx]

        # 🔥 RANDOM SHIFT
        if np.random.rand() > 0.5:
            shift_x = np.random.randint(-2, 2)
            shift_y = np.random.randint(-2, 2)
            img = np.roll(img, shift_x, axis=1)
            img = np.roll(img, shift_y, axis=2)

        # 🔥 NOISE
        if np.random.rand() > 0.7:
            img += np.random.normal(0, 0.05, img.shape)
            img = np.clip(img, 0, 1)

        return torch.tensor(img, dtype=torch.float32), label
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F


class CrowdDataset(Dataset):
    def __init__(self, img_dir, den_dir, img_size=(800, 576)):
        self.img_dir = img_dir
        self.den_dir = den_dir
        self.images = os.listdir(img_dir)
        self.img_size = img_size  # (H, W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        den_path = os.path.join(self.den_dir, img_name.replace(".jpg", ".npy"))

        # --- Load image ---
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image explicitly
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # --- Load density map ---
        density = np.load(den_path)
        density = cv2.resize(
            density,
            (self.img_size[1], self.img_size[0]),
            interpolation=cv2.INTER_LINEAR
        )

        density = torch.tensor(density, dtype=torch.float32).unsqueeze(0)

        # Downsample density map by 8 (CSRNet requirement)
        density = F.interpolate(
            density.unsqueeze(0),
            scale_factor=1/8,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return image, density

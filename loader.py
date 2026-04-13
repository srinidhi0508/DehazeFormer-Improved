import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class PairLoader(Dataset):
    def __init__(self, root_dir, mode, augment=True, patch_size=128):
        self.root_dir = root_dir
        self.mode = mode
        self.augment = augment
        self.patch_size = patch_size

        self.hazy_dir = os.path.join(root_dir, 'train', 'hazy')
        self.gt_dir = os.path.join(root_dir, 'train', 'GT')

        self.hazy_images = sorted(os.listdir(self.hazy_dir))

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_name = self.hazy_images[idx]
        hazy_path = os.path.join(self.hazy_dir, hazy_name)

        # 🔥 FIX: mapping hazy → GT
        gt_name = hazy_name.split('_')[0] + ".jpg"
        gt_path = os.path.join(self.gt_dir, gt_name)

        hazy = cv2.imread(hazy_path)
        gt = cv2.imread(gt_path)

        if hazy is None or gt is None:
            raise Exception(f"Error reading image: {hazy_name}")

        hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        hazy = hazy.astype(np.float32) / 255.0
        gt = gt.astype(np.float32) / 255.0

        # ===== Random Crop =====
        if self.augment:
            h, w, _ = hazy.shape
            ps = self.patch_size

            if h > ps and w > ps:
                x = random.randint(0, w - ps)
                y = random.randint(0, h - ps)

                hazy = hazy[y:y+ps, x:x+ps]
                gt = gt[y:y+ps, x:x+ps]

        # ===== Flip =====
        if self.augment and random.random() > 0.5:
            hazy = np.fliplr(hazy).copy()
            gt = np.fliplr(gt).copy()

        hazy = torch.from_numpy(hazy).permute(2, 0, 1)
        gt = torch.from_numpy(gt).permute(2, 0, 1)

        return hazy, gt
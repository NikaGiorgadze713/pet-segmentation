import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF

IMG_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PetSegmentation(Dataset):
    def __init__(self, root="data", split="trainval", img_size=IMG_SIZE, augment=False):
        self.base = OxfordIIITPet(
            root=root, split=split,
            target_types="segmentation", download=True,
        )
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, trimap = self.base[idx]

        image = TF.resize(image, [self.img_size, self.img_size])
        trimap = TF.resize(
            trimap, [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                trimap = TF.hflip(trimap)
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        image = TF.to_tensor(image)
        image = TF.normalize(image, MEAN, STD)

        trimap = torch.from_numpy(np.array(trimap))
        mask = (trimap != 2).float().unsqueeze(0)

        return image, mask


def get_dataloaders(root="data", batch_size=16, val_fraction=0.1, seed=42, augment=False):
    train_full = PetSegmentation(root, split="trainval", augment=augment)
    val_full = PetSegmentation(root, split="trainval", augment=False)

    n = len(train_full)
    n_val = int(n * val_fraction)
    n_train = n - n_val
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_ds = Subset(train_full, perm[:n_train])
    val_ds = Subset(val_full, perm[n_train:])

    test_ds = PetSegmentation(root, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader
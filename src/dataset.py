import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF

IMG_SIZE = 256
# Average color values of ImageNet photos (used to normalize)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PetSegmentation(Dataset):
    """Returns (image, mask) pairs ready for the model."""

    def __init__(self, root="data", split="trainval", img_size=IMG_SIZE):
        self.base = OxfordIIITPet(
            root=root, split=split,
            target_types="segmentation", download=True,
        )
        self.img_size = img_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, trimap = self.base[idx]

        # 1. Resize both to the same size
        image = TF.resize(image, [self.img_size, self.img_size])
        trimap = TF.resize(
            trimap, [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        # 2. Photo -> tensor [3, H, W] with values 0..1
        image = TF.to_tensor(image)
        # 3. Normalize
        image = TF.normalize(image, MEAN, STD)

        # 4. Mask: 1 = pet, 2 = background, 3 = border
        #    -> pet and border become 1, background becomes 0
        trimap = torch.from_numpy(np.array(trimap))
        mask = (trimap != 2).float().unsqueeze(0)  # shape [1, H, W]

        return image, mask


def get_dataloaders(root="data", batch_size=16, val_fraction=0.1, seed=42):
    full = PetSegmentation(root, split="trainval")
    n_val = int(len(full) * val_fraction)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    test_ds = PetSegmentation(root, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader
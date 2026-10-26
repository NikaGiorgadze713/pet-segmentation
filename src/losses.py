import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def forward(self, logits, masks, eps=1.0):
        probs = torch.sigmoid(logits)
        intersection = (probs * masks).sum(dim=(1, 2, 3))
        total = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice = (2 * intersection + eps) / (total + eps)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, masks):
        return self.bce(logits, masks) + self.dice(logits, masks)


def get_loss(name="bce"):
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"Unknown loss: {name}")
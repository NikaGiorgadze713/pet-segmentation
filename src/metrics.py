import torch


@torch.no_grad()
def iou_and_dice(logits, masks, threshold=0.5, eps=1e-7):
    """Average IoU and Dice over a batch."""
    preds = (torch.sigmoid(logits) > threshold).float()  # scores -> 0 or 1

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    pred_area = preds.sum(dim=(1, 2, 3))
    true_area = masks.sum(dim=(1, 2, 3))
    union = pred_area + true_area - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_area + true_area + eps)
    return iou.mean().item(), dice.mean().item()
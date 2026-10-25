import argparse
import time

import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.model import UNet
from src.metrics import iou_and_dice


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)            # 1. guess
        loss = loss_fn(logits, masks)     # 2. how wrong?

        optimizer.zero_grad()             # 3. clear old corrections
        loss.backward()                   # 4. find what caused the error
        optimizer.step()                  # 5. adjust weights a little

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        bs = images.size(0)
        total_loss += loss_fn(logits, masks).item() * bs
        iou, dice = iou_and_dice(logits, masks)
        total_iou += iou * bs
        total_dice += dice * bs
    n = len(loader.dataset)
    return total_loss / n, total_iou / n, total_dice / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="unet_scratch.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_loader, val_loader, _ = get_dataloaders(batch_size=args.batch_size)

    model = UNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_iou, val_dice = evaluate(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"IoU {val_iou:.4f} | Dice {val_dice:.4f} | {time.time() - start:.0f}s"
        )
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), args.save_path)
            print("  -> saved best model")

    print("Best val IoU:", round(best_iou, 4))


if __name__ == "__main__":
    main()
import argparse
import csv
import os
import time

import torch

from src.dataset import get_dataloaders
from src.model import get_model
from src.losses import get_loss
from src.metrics import iou_and_dice


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


def save_result(row, path="results.csv"):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unet_scratch",
                        choices=["unet_scratch", "unet_resnet34", "unet_mobilenet"])
    parser.add_argument("--loss", default="bce", choices=["bce", "bce_dice"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or f"{args.model}_{args.loss}{'_aug' if args.augment else ''}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Experiment: {name} | device: {device}")
    torch.manual_seed(42)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, augment=args.augment
    )

    model = get_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    loss_fn = get_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/{name}.pth"

    best_iou, best_dice = 0.0, 0.0
    start_all = time.time()
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
            best_iou, best_dice = val_iou, val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  saved best model to {save_path}")

    minutes = (time.time() - start_all) / 60
    print(f"Best val IoU: {best_iou:.4f}")

    save_result({
        "experiment": name,
        "model": args.model,
        "loss": args.loss,
        "augment": args.augment,
        "epochs": args.epochs,
        "lr": args.lr,
        "params_M": round(n_params / 1e6, 2),
        "best_val_iou": round(best_iou, 4),
        "best_val_dice": round(best_dice, 4),
        "train_minutes": round(minutes, 1),
    })


if __name__ == "__main__":
    main()
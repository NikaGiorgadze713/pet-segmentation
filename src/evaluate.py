import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.dataset import get_dataloaders, MEAN, STD
from src.model import get_model


@torch.no_grad()
def per_image_scores(model, loader, device, threshold=0.5, eps=1e-7):
    model.eval()
    ious, dices = [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = (torch.sigmoid(model(images)) > threshold).float()
        inter = (preds * masks).sum(dim=(1, 2, 3))
        pred_area = preds.sum(dim=(1, 2, 3))
        true_area = masks.sum(dim=(1, 2, 3))
        ious += ((inter + eps) / (pred_area + true_area - inter + eps)).tolist()
        dices += ((2 * inter + eps) / (pred_area + true_area + eps)).tolist()
    return ious, dices


@torch.no_grad()
def save_examples(model, dataset, indices, ious, path, device, title):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    fig, axes = plt.subplots(3, len(indices), figsize=(3 * len(indices), 9))
    for col, idx in enumerate(indices):
        image, mask = dataset[idx]
        logits = model(image.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits) > 0.5).float().cpu()[0, 0]
        photo = (image * std + mean).permute(1, 2, 0).clamp(0, 1)
        axes[0, col].imshow(photo)
        axes[0, col].set_title(f"IoU {ious[idx]:.2f}")
        axes[1, col].imshow(mask[0], cmap="gray")
        axes[2, col].imshow(pred, cmap="gray")
        for r in range(3):
            axes[r, col].axis("off")
    axes[1, 0].set_title("Correct mask", loc="left")
    axes[2, 0].set_title("Prediction", loc="left")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["unet_scratch", "unet_resnet34", "unet_mobilenet"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=16)
    ious, dices = per_image_scores(model, test_loader, device)
    test_iou = sum(ious) / len(ious)
    test_dice = sum(dices) / len(dices)
    print(f"{name} on test set ({len(ious)} images): IoU {test_iou:.4f} | Dice {test_dice:.4f}")

    order = sorted(range(len(ious)), key=lambda i: ious[i])
    dataset = test_loader.dataset
    save_examples(model, dataset, order[:6], ious,
                  f"{args.out_dir}/{name}_worst.png", device, f"{name}: worst predictions")
    save_examples(model, dataset, order[-6:], ious,
                  f"{args.out_dir}/{name}_best.png", device, f"{name}: best predictions")

    path = f"{args.out_dir}/test_results.csv"
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["experiment", "test_iou", "test_dice", "test_images"])
        writer.writerow([name, round(test_iou, 4), round(test_dice, 4), len(ious)])
    print(f"Saved pictures and scores to {args.out_dir}/")


if __name__ == "__main__":
    main()
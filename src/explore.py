import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import OxfordIIITPet


def pet_fraction(trimap):
    return (np.array(trimap) != 2).mean()


def main():
    os.makedirs("results", exist_ok=True)
    trainval = OxfordIIITPet("data", split="trainval",
                             target_types=["binary-category", "segmentation"], download=True)
    test = OxfordIIITPet("data", split="test", target_types="segmentation", download=True)

    species, widths, heights, fractions = Counter(), [], [], []
    for i in range(len(trainval)):
        image, (label, trimap) = trainval[i]
        species["cat" if label == 0 else "dog"] += 1
        widths.append(image.size[0])
        heights.append(image.size[1])
        fractions.append(pet_fraction(trimap))

    test_fractions = [pet_fraction(test[i][1]) for i in range(len(test))]
    empty_trainval = sum(f == 0 for f in fractions)
    empty_test = sum(f == 0 for f in test_fractions)

    lines = [
        f"trainval images: {len(trainval)}",
        f"test images: {len(test)}",
        f"cats: {species['cat']}, dogs: {species['dog']}",
        f"width min/median/max: {min(widths)} / {int(np.median(widths))} / {max(widths)}",
        f"height min/median/max: {min(heights)} / {int(np.median(heights))} / {max(heights)}",
        f"pet pixel fraction mean: {np.mean(fractions):.3f}",
        f"pet pixel fraction min/max: {min(fractions):.3f} / {max(fractions):.3f}",
        f"empty masks (no pet pixels) in trainval: {empty_trainval}",
        f"empty masks (no pet pixels) in test: {empty_test}",
        "trimap values: 1 = pet, 2 = background, 3 = border",
    ]
    with open("results/data_stats.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for col, idx in enumerate(np.linspace(0, len(trainval) - 1, 5, dtype=int)):
        image, (_, trimap) = trainval[idx]
        axes[0, col].imshow(image)
        axes[1, col].imshow(np.array(trimap), vmin=1, vmax=3)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.suptitle("Samples: photo (top) and trimap (bottom: pet, background, border)")
    plt.tight_layout()
    plt.savefig("results/data_samples.png", dpi=100)
    plt.close(fig)

    plt.figure(figsize=(7, 4))
    plt.hist(fractions, bins=30)
    plt.xlabel("Fraction of image covered by the pet")
    plt.ylabel("Number of images")
    plt.title("How much of each photo is the pet")
    plt.tight_layout()
    plt.savefig("results/pet_fraction_hist.png", dpi=100)


if __name__ == "__main__":
    main()
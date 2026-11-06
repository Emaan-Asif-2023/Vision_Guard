import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Dataset folders
DATASET_DIRS = {
    "AI-generated": r"C:\Users\hp\Downloads\archive\AiArtData\AiArtData",
    "Real": r"C:\Users\hp\Downloads\archive\RealArt\RealArt",
}

random.seed(42)

fig, axes = plt.subplots(2, 5, figsize=(15, 7))

for row, (class_name, folder) in enumerate(DATASET_DIRS.items()):
    image_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    selected = random.sample(image_files, min(5, len(image_files)))

    for col, image_path in enumerate(selected):
        with Image.open(image_path) as image:
            axes[row, col].imshow(image.convert("RGB"))
        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_ylabel(
                class_name, fontsize=12, rotation=90, labelpad=15
            )

    for col in range(len(selected), 5):
        axes[row, col].axis("off")

plt.suptitle("Sample Images: AI-generated vs Real", fontsize=16)
plt.tight_layout()
plt.savefig("sample_image_grid.png", dpi=200, bbox_inches="tight")
plt.show()

print("Saved: sample_image_grid.png")


# Image width histograms
plt.figure(figsize=(10, 6))

for class_name, folder in DATASET_DIRS.items():
    widths = []

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            image_path = os.path.join(folder, filename)

            try:
                with Image.open(image_path) as image:
                    widths.append(image.width)
            except Exception:
                print(f"Skipping unreadable image: {image_path}")

    plt.hist(widths, bins=30, alpha=0.6, label=class_name)

plt.title("Image Width Distribution: AI-generated vs Real")
plt.xlabel("Width (pixels)")
plt.ylabel("Number of Images")
plt.legend()
plt.tight_layout()
plt.savefig("width_histogram.png", dpi=200)
plt.show()

print("Saved: width_histogram.png")


# Image height histograms
plt.figure(figsize=(10, 6))

for class_name, folder in DATASET_DIRS.items():
    heights = []

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            image_path = os.path.join(folder, filename)

            try:
                with Image.open(image_path) as image:
                    heights.append(image.height)
            except Exception:
                print(f"Skipping unreadable image: {image_path}")

    plt.hist(heights, bins=30, alpha=0.6, label=class_name)

plt.title("Image Height Distribution: AI-generated vs Real")
plt.xlabel("Height (pixels)")
plt.ylabel("Number of Images")
plt.legend()
plt.tight_layout()
plt.savefig("height_histogram.png", dpi=200)
plt.show()

print("Saved: height_histogram.png")


# File size histograms
import numpy as np

plt.figure(figsize=(10, 6))

for class_name, folder in DATASET_DIRS.items():
    file_sizes = []

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            image_path = os.path.join(folder, filename)

            try:
                file_sizes.append(os.path.getsize(image_path) / 1024)
            except OSError:
                print(f"Skipping unreadable file: {image_path}")

    plt.hist(file_sizes, bins=30, alpha=0.6, label=class_name)

plt.title("File Size Distribution: AI-generated vs Real")
plt.xlabel("File Size (KB)")
plt.ylabel("Number of Images")
plt.legend()
plt.tight_layout()
plt.savefig("file_size_histogram.png", dpi=200)
plt.show()

print("Saved: file_size_histogram.png")

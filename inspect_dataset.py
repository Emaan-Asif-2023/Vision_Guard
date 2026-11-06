import os
import csv
import hashlib
from collections import Counter
from PIL import Image, UnidentifiedImageError

DATASET_DIRS = {
    "AiArtData": r"C:\Users\hp\Downloads\archive\AiArtData\AiArtData",
    "RealArt": r"C:\Users\hp\Downloads\archive\RealArt\RealArt",
}

OUTPUT_CSV = "dataset_audit.csv"
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".webp", ".tif", ".tiff", ".jfif"
}

records = []
unreadable_files = []
hash_to_files = {}

for class_name, folder in DATASET_DIRS.items():
    if not os.path.isdir(folder):
        print(f"ERROR: Folder not found: {folder}")
        continue

    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            if os.path.splitext(filename)[1].lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                file_size = os.path.getsize(file_path)

                # Calculate SHA-256 hash
                sha256 = hashlib.sha256()
                with open(file_path, "rb") as file:
                    for chunk in iter(lambda: file.read(1024 * 1024), b""):
                        sha256.update(chunk)
                file_hash = sha256.hexdigest()

                hash_to_files.setdefault(file_hash, []).append(file_path)

                # Read image metadata and verify the image
                with Image.open(file_path) as image:
                    image_format = image.format
                    color_mode = image.mode
                    width, height = image.size
                    image.verify()

                records.append({
                    "class": class_name,
                    "file_name": filename,
                    "file_path": file_path,
                    "file_size_bytes": file_size,
                    "width": width,
                    "height": height,
                    "color_mode": color_mode,
                    "format": image_format,
                    "sha256": file_hash,
                    "status": "OK",
                })

            except (OSError, ValueError, SyntaxError, UnidentifiedImageError) as error:
                unreadable_files.append({
                    "class": class_name,
                    "file_path": file_path,
                    "error": str(error),
                })

# Save image-level audit table
fieldnames = [
    "class", "file_name", "file_path", "file_size_bytes",
    "width", "height", "color_mode", "format", "sha256", "status"
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

# Find exact duplicates by SHA-256
duplicate_groups = [
    paths for paths in hash_to_files.values() if len(paths) > 1
]
duplicate_file_count = sum(len(paths) - 1 for paths in duplicate_groups)

with open("unreadable_files.csv", "w", newline="", encoding="utf-8-sig") as csvfile:
    fields = ["class", "file_path", "error"]
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(unreadable_files)

with open("duplicate_files.csv", "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["sha256", "file_path"])
    for file_hash, paths in hash_to_files.items():
        if len(paths) > 1:
            for path in paths:
                writer.writerow([file_hash, path])

# Report class counts and summary statistics
print("\n========== DATASET AUDIT ==========")
print(f"Readable images: {len(records)}")
print(f"Unreadable image files: {len(unreadable_files)}")
print(f"Duplicate groups: {len(duplicate_groups)}")
print(f"Extra duplicate copies: {duplicate_file_count}")

for class_name in DATASET_DIRS:
    class_records = [r for r in records if r["class"] == class_name]
    print(f"\nClass: {class_name}")
    print(f"Readable image count: {len(class_records)}")

    for field, label in [
        ("width", "Width"),
        ("height", "Height"),
        ("file_size_bytes", "File size (bytes)"),
    ]:
        values = sorted(r[field] for r in class_records)
        if values:
            middle = len(values) // 2
            if len(values) % 2:
                median = values[middle]
            else:
                median = (values[middle - 1] + values[middle]) / 2

            print(
                f"{label}: min={values[0]}, "
                f"median={median}, max={values[-1]}"
            )

print("\nColor mode counts:")
print(Counter(r["color_mode"] for r in records))

print("\nImage format counts:")
print(Counter(r["format"] for r in records))

print("\nCreated files:")
print(f"- {OUTPUT_CSV}")
print("- unreadable_files.csv")
print("- duplicate_files.csv")
print("\nAudit complete.")

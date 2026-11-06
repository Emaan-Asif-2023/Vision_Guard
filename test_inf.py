
from pathlib import Path
from inference import predict_image

def first_image(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in Path(folder).rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            return path
    raise FileNotFoundError(f"No image found in {folder}")

for folder, expected in [
    ("dataset/AI", "AI-generated"),
    ("dataset/Real", "Real"),
]:
    image_path = first_image(folder)

    with open(image_path, "rb") as image_file:
        result = predict_image(image_file)

    print(f"\nKnown category: {expected}")
    print(f"Image: {image_path}")
    print(f"Predicted label: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"AI probability: {result['ai_probability']:.2%}")
    print(f"Caveat: {result['caveat']}")

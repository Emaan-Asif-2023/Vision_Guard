
import mlflow
import dagshub
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

dagshub.init(
    repo_owner="Emaan-Asif-2023",
    repo_name="my-first-repo",
    mlflow=True
)

MODEL_URI = "models:/VisionGuard_MobileNetV2@champion"

# Change this path to an image that exists in your dataset
IMAGE_PATH = "dataset/AI"

# Find one image inside the selected folder
if Path(IMAGE_PATH).is_dir():
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [
        p for p in Path(IMAGE_PATH).rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    if not images:
        raise FileNotFoundError(
            f"No supported images found in {IMAGE_PATH}"
        )
    IMAGE_PATH = str(images[0])

if not Path(IMAGE_PATH).is_file():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

print("Loading champion model...")
model = mlflow.keras.load_model(MODEL_URI)

# Match the model's expected 224x224 RGB input and [-1, 1] scaling
image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=(224, 224),
    color_mode="rgb"
)
image = tf.keras.utils.img_to_array(image)
image = image / 127.5 - 1.0
image = np.expand_dims(image, axis=0)

probability = float(model.predict(image, verbose=0)[0][0])

# Training labels: AI = 1, Real = 0
prediction = "AI-generated" if probability >= 0.5 else "Real"

print("\nPrediction successful!")
print("Image:", IMAGE_PATH)
print("Predicted class:", prediction)
print(f"AI probability: {probability:.4f}")
print("Model URI:", MODEL_URI)

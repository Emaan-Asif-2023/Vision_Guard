
import os
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/champion.keras"
)

THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

# Load the model once when this module is imported.
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_file):
    """
    Accept a file-like image object and return its label,
    confidence, and AI probability.

    Training preprocessing: RGB, 224x224, pixel values
    scaled from [0, 255] to [-1, 1].
    """
    try:
        image_file.seek(0)
        with Image.open(image_file) as img:
            img = img.convert("RGB")
            img = img.resize((224, 224))
            array = np.asarray(img, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    # Match MobileNetV2 training preprocessing.
    array = array / 127.5 - 1.0
    array = np.expand_dims(array, axis=0)

    ai_probability = float(model.predict(array, verbose=0)[0][0])

    if ai_probability >= THRESHOLD:
        label = "AI-generated"
        confidence = ai_probability
    else:
        label = "Real"
        confidence = 1.0 - ai_probability

    return {
        "label": label,
        "confidence": confidence,
        "ai_probability": ai_probability,
        "threshold": THRESHOLD,
        "caveat": (
            "This is a model estimate, not proof of an image's origin. "
            "Misclassifications are possible."
        )
    }

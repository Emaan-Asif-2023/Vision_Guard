import os
import requests


def predict_image(image_file):

    api_url = os.getenv("MODEL_API_URL")
    api_key = os.getenv("MODEL_API_KEY")

    if not api_url or not api_key:
        raise RuntimeError("Model API configuration is missing.")

    image_file.seek(0)
    file_bytes = image_file.read()
    
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        headers={"X-API-Key": api_key},
        files={
            "file": (
                image_file.filename or "image.jpg",
                file_bytes,
                image_file.mimetype or "image/jpeg"
            )
        },
        timeout=60
    )

    response.raise_for_status()

    return response.json()
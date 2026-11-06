# VisionGuard: AI vs Real Image Classification

VisionGuard is a deep learning application that classifies images as AI-generated or real using a fine-tuned MobileNetV2 model.

## Features
- AI-generated vs real image classification
- Flask web application
- Prediction history and database logging
- PostgreSQL database
- Docker and Docker Compose deployment
- MLflow experiment tracking with DagsHub

## Technologies Used
- Python, TensorFlow and Keras
- Flask
- MLflow and DagsHub
- PostgreSQL
- Docker and Docker Compose
- GitHub Actions

## Run with Docker Compose

1. Clone this repository.
2. Create a `.env` file using `.env.example` as a template.
3. Build and start the application:

```bash
docker compose up --build -d
```

4. Open the application at:

`http://localhost:5001`

5. Check application health at:

`http://localhost:5001/health`

## Model
The application uses the fine-tuned MobileNetV2 model stored at `models/champion.keras`.

## Author
Emaan Asif
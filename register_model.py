
import mlflow
import mlflow.keras
import dagshub
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv()

dagshub.init(
    repo_owner="Emaan-Asif-2023",
    repo_name="my-first-repo",
    mlflow=True
)

mlflow.set_experiment("VisionGuard_Image_Classification")

model_path = "best_mobilenetv2_finetuned.keras"
model_name = "VisionGuard_MobileNetV2"

print("Loading fine-tuned MobileNetV2...")
model = load_model(model_path)

with mlflow.start_run(
    run_name="Register_Fine_Tuned_MobileNetV2"
) as run:

    print("Logging model in MLflow format...")

    model_info = mlflow.keras.log_model(
        model,
        name="model"
    )

    print("Registering model...")

    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name
    )

    print("\nRegistration successful!")
    print("Model name:", registered_model.name)
    print("Model version:", registered_model.version)

print("\nTask 3.6 registration step completed.")

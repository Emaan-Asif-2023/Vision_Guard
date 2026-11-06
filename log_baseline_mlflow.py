
import json
import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

dagshub.init(
    repo_owner="Emaan-Asif-2023",
    repo_name="my-first-repo",
    mlflow=True
)

mlflow.set_experiment("VisionGuard_Image_Classification")

with open("baseline_history.json", "r") as file:
    history = json.load(file)

with mlflow.start_run(run_name="Baseline_CNN_Existing_Results"):
    mlflow.log_params({
        "model": "Custom_CNN",
        "architecture": "Conv2D_BatchNorm_GlobalAveragePooling",
        "epochs_completed": len(history["accuracy"]),
        "learning_rate_initial": history["learning_rate"][0],
        "seed": 42,
        "results_source": "Previously completed training"
    })

    for epoch in range(len(history["accuracy"])):
        metrics = {
            key: float(values[epoch])
            for key, values in history.items()
        }
        mlflow.log_metrics(metrics, step=epoch + 1)

    mlflow.log_metric(
        "best_val_accuracy",
        max(history["val_accuracy"])
    )

    mlflow.log_artifact("baseline_history.json")
    mlflow.log_artifact("best_baseline_cnn.keras")

    mlflow.set_tag("stage", "baseline_cnn")
    mlflow.set_tag("training_reperformed", "false")

print("Baseline CNN history and model logged successfully!")

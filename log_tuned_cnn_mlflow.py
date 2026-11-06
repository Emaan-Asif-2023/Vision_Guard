
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

with open("best_hyperparameters.json", "r") as file:
    params = json.load(file)

with mlflow.start_run(run_name="Tuned_CNN_KerasTuner_Results"):
    mlflow.log_params({
        "model": "Custom_CNN_KerasTuner",
        "filters_1": params["filters_1"],
        "filters_2": params["filters_2"],
        "dense_units": params["dense_units"],
        "dropout": params["dropout"],
        "learning_rate": params["learning_rate"],
        "epochs_per_trial": 5,
        "tuning_method": "RandomSearch",
        "trials": 4,
        "results_source": "Previously completed tuning"
    })

    mlflow.log_metric("best_val_accuracy", 0.6939)

    mlflow.log_artifact("best_hyperparameters.json")
    mlflow.log_artifact("best_tuned_cnn.keras")

    mlflow.set_tag("stage", "hyperparameter_tuning")
    mlflow.set_tag("training_reperformed", "false")

print("Tuned CNN configuration and model logged successfully!")

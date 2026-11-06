
import json
import pandas as pd
import matplotlib.pyplot as plt
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

# Read actual best validation accuracies from training histories
def best_accuracy(history_file):
    with open(history_file, "r") as f:
        history = json.load(f)
    return max(history["val_accuracy"]) * 100

results = [
    {
        "Experiment": "Baseline CNN",
        "Change": "Original CNN",
        "Best Validation Accuracy (%)":
            best_accuracy("baseline_history.json")
    },
    {
        "Experiment": "Keras Tuner CNN",
        "Change": "Tuned architecture and hyperparameters",
        "Best Validation Accuracy (%)": 69.39
    },
    {
        "Experiment": "MobileNetV2 Transfer Learning",
        "Change": "Frozen pretrained backbone",
        "Best Validation Accuracy (%)":
            best_accuracy("mobilenetv2_history.json")
    },
    {
        "Experiment": "MobileNetV2 Fine-Tuning",
        "Change": "Unfroze last 30 backbone layers",
        "Best Validation Accuracy (%)":
            best_accuracy("finetuned_history.json")
    },
    {
        "Experiment": "Baseline CNN LR 0.0005",
        "Change": "Learning rate changed to 0.0005",
        "Best Validation Accuracy (%)":
            best_accuracy("baseline_lr_0005_history.json")
    },
    {
        "Experiment": "Baseline CNN Dropout 0.2",
        "Change": "Dropout reduced from 0.4 to 0.2",
        "Best Validation Accuracy (%)":
            best_accuracy("baseline_dropout_0002_history.json")
    }
]

df = pd.DataFrame(results)
df = df.sort_values(
    "Best Validation Accuracy (%)",
    ascending=False
)

df.to_csv("experiment_comparison.csv", index=False)

plt.figure(figsize=(11, 6))
bars = plt.bar(
    df["Experiment"],
    df["Best Validation Accuracy (%)"]
)
plt.ylabel("Best Validation Accuracy (%)")
plt.xlabel("Experiment")
plt.title("VisionGuard: Experiment Comparison")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 100)

for bar, value in zip(
    bars, df["Best Validation Accuracy (%)"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=8
    )

plt.tight_layout()
plt.savefig("experiment_comparison.png", dpi=150)
plt.close()

print("\nExperiment comparison:")
print(df.to_string(index=False))

with mlflow.start_run(run_name="VisionGuard_Experiment_Comparison"):
    mlflow.log_artifact("experiment_comparison.csv")
    mlflow.log_artifact("experiment_comparison.png")
    mlflow.log_param("number_of_experiments_compared", len(df))
    mlflow.set_tag(
        "purpose",
        "Compare model architectures and hyperparameter variations"
    )

print("\nComparison CSV and chart logged to DagsHub.")

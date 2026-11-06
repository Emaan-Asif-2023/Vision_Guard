
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mlflow
import dagshub
from dotenv import load_dotenv
from sklearn.metrics import roc_curve, roc_auc_score

from input_pipeline import create_dataset


# Connect to DagsHub
load_dotenv()

dagshub.init(
    repo_owner="Emaan-Asif-2023",
    repo_name="my-first-repo",
    mlflow=True,
)

mlflow.set_experiment("VisionGuard_Image_Classification")


# Load the fine-tuned model and test dataset
model = tf.keras.models.load_model(
    "best_mobilenetv2_finetuned.keras"
)

test_ds = create_dataset("test.csv", training=False)


# Apply MobileNetV2 preprocessing
def preprocess_for_mobilenet(images, labels):
    return images * 2.0 - 1.0, labels


test_ds = test_ds.map(
    preprocess_for_mobilenet,
    num_parallel_calls=tf.data.AUTOTUNE,
)

# Actual labels and predicted probabilities
test_df = pd.read_csv("test.csv")
y_true = test_df["label"].to_numpy()
y_scores = model.predict(test_ds).ravel()

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc_score = roc_auc_score(y_true, y_scores)

# Plot the ROC curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"MobileNetV2 (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("VisionGuard ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# Log the ROC curve and AUC to MLflow
with mlflow.start_run(run_name="MobileNetV2_ROC_Evaluation"):
    mlflow.log_param("model", "MobileNetV2_Fine_Tuned")
    mlflow.log_param("dataset", "test.csv")
    mlflow.log_metric("test_roc_auc", float(auc_score))
    mlflow.set_tag("stage", "roc_evaluation")
    mlflow.log_artifact("roc_curve.png")

print(f"ROC-AUC: {auc_score:.4f}")
print("Saved and logged roc_curve.png")

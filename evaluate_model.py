
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
from dotenv import load_dotenv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from input_pipeline import create_dataset


# Load environment variables and connect to DagsHub
load_dotenv()

dagshub.init(
    repo_owner="Emaan-Asif-2023",
    repo_name="my-first-repo",
    mlflow=True,
)

mlflow.set_experiment("VisionGuard_Image_Classification")


# Load the saved fine-tuned model
model = tf.keras.models.load_model(
    "best_mobilenetv2_finetuned.keras"
)

# Load the test dataset without augmentation
test_ds = create_dataset("test.csv", training=False)


# Apply MobileNetV2 preprocessing: [0, 1] to [-1, 1]
def preprocess_for_mobilenet(images, labels):
    images = images * 2.0 - 1.0
    return images, labels


test_ds = test_ds.map(
    preprocess_for_mobilenet,
    num_parallel_calls=tf.data.AUTOTUNE,
)

# Get actual labels
test_df = pd.read_csv("test.csv")
y_true = test_df["label"].to_numpy()

# Predict probabilities and class labels
probabilities = model.predict(test_ds).ravel()
y_pred = (probabilities >= 0.5).astype(int)


# Start one MLflow run for the complete evaluation
with mlflow.start_run(
    run_name="MobileNetV2_Final_Test_Evaluation"
):
    mlflow.log_param("model", "MobileNetV2_Fine_Tuned")
    mlflow.log_param("dataset", "test.csv")
    mlflow.log_param("threshold", 0.5)
    mlflow.log_param("seed", 42)
    mlflow.set_tag("stage", "final_test_evaluation")

    # Calculate and display metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    print("\n===== TEST SET RESULTS =====")
    print("Test images:", len(y_true))
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Log metrics to MLflow
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1,
    })

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Real", "AI-generated"],
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    # Generate and save the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "AI-generated"],
        yticklabels=["Real", "AI-generated"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("VisionGuard Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig("test_confusion_matrix.png", dpi=150)
    plt.close()

    # Log evaluation artifacts
    mlflow.log_artifact("test_confusion_matrix.png")
    mlflow.log_artifact("test.csv")

    # Save individual predictions for later analysis
    predictions_df = test_df.copy()
    predictions_df["probability_ai"] = probabilities
    predictions_df["predicted_label"] = y_pred
    predictions_df["correct"] = (
        predictions_df["label"] == predictions_df["predicted_label"]
    )
    predictions_df.to_csv("test_predictions.csv", index=False)
    mlflow.log_artifact("test_predictions.csv")

    print("\nSaved and logged:")
    print("- test_confusion_matrix.png")
    print("- classification_report.txt")
    print("- test.csv")
    print("- test_predictions.csv")

print("\nMLflow evaluation run completed.")

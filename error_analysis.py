
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math

from input_pipeline import create_dataset
from sklearn.metrics import confusion_matrix

# Load the fine-tuned model
model = tf.keras.models.load_model(
    "best_mobilenetv2_finetuned.keras"
)

# Load test data without augmentation
test_df = pd.read_csv("test.csv")
test_ds = create_dataset("test.csv", training=False)

# Apply MobileNetV2 preprocessing
def preprocess_for_mobilenet(images, labels):
    return images * 2.0 - 1.0, labels

test_ds = test_ds.map(
    preprocess_for_mobilenet,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Predict probabilities
probabilities = model.predict(test_ds, verbose=1).ravel()
predictions = (probabilities >= 0.5).astype(int)
actual = test_df["label"].to_numpy()

# Identify incorrectly classified images
wrong_indices = np.where(predictions != actual)[0]

print("\n===== ERROR ANALYSIS =====")
print("Total test images:", len(actual))
print("Incorrect predictions:", len(wrong_indices))
print("Correct predictions:", len(actual) - len(wrong_indices))

# Count types of errors
false_positives = np.sum((actual == 0) & (predictions == 1))
false_negatives = np.sum((actual == 1) & (predictions == 0))

print("Real images incorrectly predicted as AI:", false_positives)
print("AI images incorrectly predicted as Real:", false_negatives)

# Save details of incorrect predictions
error_df = test_df.iloc[wrong_indices].copy()
error_df["predicted_label"] = predictions[wrong_indices]
error_df["ai_probability"] = probabilities[wrong_indices]
error_df["actual_class"] = error_df["label"].map(
    {0: "Real", 1: "AI-generated"}
)
error_df["predicted_class"] = error_df["predicted_label"].map(
    {0: "Real", 1: "AI-generated"}
)
error_df.to_csv("misclassified_images.csv", index=False)
print("Saved misclassified_images.csv")

# Display misclassified images
if len(wrong_indices) > 0:
    max_images = min(len(wrong_indices), 20)
    columns = 4
    rows = math.ceil(max_images / columns)

    plt.figure(figsize=(14, rows * 3.5))

    for position, index in enumerate(wrong_indices[:max_images]):
        image_path = test_df.iloc[index]["filepath"]
        image = tf.keras.utils.load_img(
            image_path, target_size=(224, 224)
        )

        actual_label = "AI-generated" if actual[index] == 1 else "Real"
        predicted_label = (
            "AI-generated" if predictions[index] == 1 else "Real"
        )

        plt.subplot(rows, columns, position + 1)
        plt.imshow(image)
        plt.title(
            f"Actual: {actual_label}\n"
            f"Predicted: {predicted_label}\n"
            f"AI probability: {probabilities[index]:.2f}",
            fontsize=9
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("misclassified_grid.png", dpi=150)
    plt.show()

    print("Saved misclassified_grid.png")
else:
    print("No misclassified images found.")

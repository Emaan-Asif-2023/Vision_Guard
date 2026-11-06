
import numpy as np
import tensorflow as tf
from input_pipeline import create_dataset

val_ds = create_dataset("val.csv")
model = tf.keras.models.load_model("best_baseline_cnn.keras")

probabilities = model.predict(val_ds, verbose=0).flatten()
actual_labels = np.concatenate([
    labels.numpy() for _, labels in val_ds
]).astype(int)

predicted_labels = (probabilities >= 0.5).astype(int)

print("\nValidation diagnosis")
print("--------------------")
print("Actual class counts (Real=0, AI=1):")
print(np.bincount(actual_labels, minlength=2))

print("\nPredicted class counts (Real=0, AI=1):")
print(np.bincount(predicted_labels, minlength=2))

print("\nPrediction probabilities:")
print("Minimum:", round(float(probabilities.min()), 4))
print("Maximum:", round(float(probabilities.max()), 4))
print("Mean:", round(float(probabilities.mean()), 4))

print("\nFirst 10 predictions:")
for actual, probability, predicted in zip(
    actual_labels[:10],
    probabilities[:10],
    predicted_labels[:10]
):
    print(
        f"Actual: {actual}, "
        f"Probability of AI: {probability:.4f}, "
        f"Predicted: {predicted}"
    )

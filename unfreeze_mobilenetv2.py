
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from input_pipeline import create_dataset
import mlflow
import dagshub
from dotenv import load_dotenv
import os
import subprocess

load_dotenv()

dagshub.init(
    repo_owner="Emaan-Asif-2023",
    repo_name="my-first-repo",
    mlflow=True
)

mlflow.set_experiment("VisionGuard_Image_Classification")

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Load training and validation datasets
train_ds = create_dataset("train.csv", training=True)
val_ds = create_dataset("val.csv")

# Apply MobileNetV2 preprocessing: [0, 1] to [-1, 1]
def preprocess_for_mobilenet(images, labels):
    images = images * 2.0 - 1.0
    return images, labels

train_ds = train_ds.map(
    preprocess_for_mobilenet,
    num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_ds.map(
    preprocess_for_mobilenet,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Load the best model from transfer learning
model = tf.keras.models.load_model("best_mobilenetv2.keras")

# Find the pretrained MobileNetV2 base model
base_model = next(
    layer for layer in model.layers
    if isinstance(layer, tf.keras.Model)
)

# Unfreeze the base model, then freeze its earlier layers
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Keep Batch Normalization layers frozen for stability
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print("Total MobileNetV2 layers:", len(base_model.layers))
print("Fine-tuning from layer:", fine_tune_at)
print("Trainable parameters:", sum(
    tf.keras.backend.count_params(w)
    for w in model.trainable_weights
))

# Use a very small learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)
class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, metric_value in logs.items():
            mlflow.log_metric(
                metric_name,
                float(metric_value),
                step=epoch + 1
            )

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_mobilenetv2_finetuned.keras",
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-7
    )
]

with mlflow.start_run(run_name="MobileNetV2_Fine_Tuning"):
    mlflow.log_param("model", "MobileNetV2")
    mlflow.log_param("transfer_learning", True)
    mlflow.log_param("fine_tuning", True)
    mlflow.log_param("unfrozen_layers", 30)
    mlflow.log_param("learning_rate", 0.00001)
    mlflow.log_param("epochs", 5)
    mlflow.log_param("seed", SEED)

    mlflow.set_tag("stage", "fine_tuning")

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()
        mlflow.set_tag("git_commit", git_commit)
    except Exception:
        mlflow.set_tag("git_commit", "unavailable")

    if os.path.exists("dataset.dvc"):
        mlflow.log_artifact("dataset.dvc")
        mlflow.set_tag("dvc_version", "dataset.dvc")

    callbacks.append(MLflowMetricsCallback())

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks
    )

    run_id = mlflow.active_run().info.run_id

# Save training history
with open("finetuned_history.json", "w") as f:
    json.dump(history.history, f)

# Plot loss
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 Fine-Tuning Loss")
plt.legend()
plt.savefig("finetuned_loss.png", dpi=150)
plt.close()

# Plot accuracy
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Fine-Tuning Accuracy")
plt.legend()
plt.savefig("finetuned_accuracy.png", dpi=150)
plt.close()

best_epoch = min(
    range(len(history.history["val_loss"])),
    key=lambda i: history.history["val_loss"][i]
)

print("\nFine-tuning complete.")
print("Best epoch:", best_epoch + 1)
print(
    "Best validation accuracy:",
    f"{history.history['val_accuracy'][best_epoch] * 100:.2f}%"
)
print(
    "Best validation loss:",
    f"{history.history['val_loss'][best_epoch]:.4f}"
)
print("Saved best_mobilenetv2_finetuned.keras")
print("Saved finetuned_history.json and training curves.")
with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact("finetuned_history.json")
    mlflow.log_artifact("finetuned_loss.png")
    mlflow.log_artifact("finetuned_accuracy.png")
    mlflow.log_artifact("best_mobilenetv2_finetuned.keras")

print("Fine-tuning metrics and artifacts logged to MLflow.")
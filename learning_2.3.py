
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
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
IMG_SIZE = (224, 224)

tf.keras.utils.set_random_seed(SEED)

# Load the datasets
train_ds = create_dataset("train.csv", training=True)
val_ds = create_dataset("val.csv")

# MobileNetV2 requires pixel values scaled from [0, 1] to [-1, 1].
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

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# Build the classification model
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_mobilenetv2.keras",
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]

class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for metric_name, metric_value in logs.items():
            mlflow.log_metric(
                metric_name,
                float(metric_value),
                step=epoch + 1
            )


with mlflow.start_run(run_name="MobileNetV2_Transfer_Learning"):

    mlflow.log_param("model", "MobileNetV2")
    mlflow.log_param("transfer_learning", True)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("seed", 42)
    

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


    mlflow.set_tag("stage", "transfer_learning")
    run_id = mlflow.active_run().info.run_id

    callbacks.append(MLflowMetricsCallback())

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks
    )


with open("mobilenetv2_history.json", "w") as f:
    json.dump(history.history, f)


plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 - Loss")
plt.legend()
plt.tight_layout()
plt.savefig("mobilenetv2_loss.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 - Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("mobilenetv2_accuracy.png", dpi=150)
plt.close()

best_epoch = min(
    range(len(history.history["val_loss"])),
    key=lambda i: history.history["val_loss"][i]
) + 1

print(f"Best epoch based on validation loss: {best_epoch}")
print("Saved best_mobilenetv2.keras")
print("Saved mobilenetv2_history.json and training curves")


with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact("mobilenetv2_history.json")
    mlflow.log_artifact("mobilenetv2_loss.png")
    mlflow.log_artifact("mobilenetv2_accuracy.png")
    mlflow.log_artifact("best_mobilenetv2.keras")

print("MLflow metrics and artifacts logged successfully.")


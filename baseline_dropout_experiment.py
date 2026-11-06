
import os
import json
import subprocess
import numpy as np
import tensorflow as tf
import mlflow
import dagshub
from dotenv import load_dotenv
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

load_dotenv()
dagshub.init(
    repo_owner="Emaan-Asif-2023",
    repo_name="my-first-repo",
    mlflow=True
)
mlflow.set_experiment("VisionGuard_Image_Classification")

SEED = 42
BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 0.0001
DROPOUT = 0.2

tf.keras.utils.set_random_seed(SEED)

# Load the existing dataset splits
def load_data(csv_file, training=False):
    import pandas as pd

    df = pd.read_csv(csv_file)
    paths = df["filepath"].astype(str).tolist()
    labels = df["label"].astype(np.float32).tolist()

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(
            image, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = load_data("train.csv", training=True)
val_ds = load_data("val.csv")

# Build the CNN with reduced dropout
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(DROPOUT),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for key, value in (logs or {}).items():
            if value is not None:
                mlflow.log_metric(key, float(value), step=epoch + 1)

run_id = None
with mlflow.start_run(run_name="Baseline_CNN_Dropout_020") as run:
    run_id = run.info.run_id

    mlflow.log_params({
        "model": "Baseline CNN",
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seed": SEED,
        "image_size": "224x224",
        "optimizer": "Adam"
    })

    mlflow.set_tags({
        "stage": "baseline_experiment",
        "experiment_variation": "dropout_0.2"
    })

    try:
        mlflow.set_tag(
            "git_commit",
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True
            ).strip()
        )
    except Exception:
        mlflow.set_tag("git_commit", "unavailable")

    mlflow.log_artifact("dataset.dvc")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            MLflowMetricsCallback(),
            EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    history_dict = {
        key: [float(x) for x in values]
        for key, values in history.history.items()
    }

    with open("baseline_dropout_0002_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    model.save("baseline_dropout_0002.keras")

    mlflow.log_artifact("baseline_dropout_0002_history.json")
    mlflow.log_artifact("baseline_dropout_0002.keras")

    best_val_accuracy = max(history.history["val_accuracy"])
    best_val_loss = min(history.history["val_loss"])

    mlflow.log_metric("best_val_accuracy", float(best_val_accuracy))
    mlflow.log_metric("best_val_loss", float(best_val_loss))

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"MLflow run ID: {run_id}")

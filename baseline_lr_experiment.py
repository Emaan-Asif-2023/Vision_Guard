
import json
import mlflow
import dagshub
import tensorflow as tf
from dotenv import load_dotenv
from input_pipeline import create_dataset

load_dotenv()

dagshub.init(
    repo_owner="Emaan-Asif-2023",
    repo_name="my-first-repo",
    mlflow=True
)
mlflow.set_experiment("VisionGuard_Image_Classification")

SEED = 42
tf.keras.utils.set_random_seed(SEED)

train_ds = create_dataset("train.csv", training=True)
val_ds = create_dataset("val.csv", training=False)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

class LogMetricsToMLflow(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metrics(
                {key: float(value) for key, value in logs.items()},
                step=epoch + 1
            )

with mlflow.start_run(run_name="Baseline_CNN_LR_0005"):
    mlflow.log_params({
        "model": "Custom_CNN",
        "learning_rate": 0.0005,
        "dropout": 0.4,
        "image_size": 224,
        "batch_size": 16,
        "seed": SEED,
        "epochs_requested": 6
    })

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=6,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            ),
            LogMetricsToMLflow()
        ]
    )

    model.save("baseline_lr_0005.keras")
    with open("baseline_lr_0005_history.json", "w") as f:
        json.dump(history.history, f)

    mlflow.log_artifact("baseline_lr_0005.keras")
    mlflow.log_artifact("baseline_lr_0005_history.json")
    mlflow.set_tag("stage", "learning_rate_comparison")

    mlflow.log_metric(
        "best_val_accuracy",
        float(max(history.history["val_accuracy"]))
    )

print("Learning-rate experiment completed and logged.")

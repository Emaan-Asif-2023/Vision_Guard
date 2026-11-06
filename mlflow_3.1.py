
import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
import dagshub
from dotenv import load_dotenv

load_dotenv()
SEED = 42
tf.keras.utils.set_random_seed(SEED)

dagshub.init(
    repo_owner="Emaan-Asif-2023",
    repo_name="my-first-repo",
    mlflow=True
)
mlflow.set_experiment("VisionGuard_MLflow_Toy_Lab")


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = x_train[:10000].astype("float32") / 255.0
y_train = y_train[:10000]
x_test = x_test[:2000].astype("float32") / 255.0
y_test = y_test[:2000]

learning_rates = [0.001, 0.0001]

for lr in learning_rates:
    with mlflow.start_run(run_name=f"MNIST_lr_{lr}") as run:
        mlflow.set_tag("dataset", "MNIST")
        mlflow.set_tag("purpose", "Task 3.1 local MLflow tool lab")
        mlflow.set_tag("seed", str(SEED))

        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", 128)
        mlflow.log_param("epochs", 3)
        mlflow.log_param("training_samples", len(x_train))
        mlflow.log_param("test_samples", len(x_test))

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = model.fit(
            x_train, y_train,
            validation_split=0.2,
            epochs=3,
            batch_size=128,
            verbose=1
        )

        test_loss, test_accuracy = model.evaluate(
            x_test, y_test, verbose=0
        )

        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_accuracy))

        # Log each epoch's training and validation metrics
        for epoch in range(len(history.history["loss"])):
            mlflow.log_metric(
                "train_loss", float(history.history["loss"][epoch]),
                step=epoch + 1
            )
            mlflow.log_metric(
                "train_accuracy",
                float(history.history["accuracy"][epoch]),
                step=epoch + 1
            )
            mlflow.log_metric(
                "val_loss",
                float(history.history["val_loss"][epoch]),
                step=epoch + 1
            )
            mlflow.log_metric(
                "val_accuracy",
                float(history.history["val_accuracy"][epoch]),
                step=epoch + 1
            )

        # Save and log a training plot
        plt.figure()
        plt.plot(history.history["accuracy"], label="Training accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"MNIST Accuracy (learning rate={lr})")
        plt.legend()
        plt.tight_layout()
        plot_path = f"toy_accuracy_lr_{lr}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(plot_path)

        mlflow.tensorflow.log_model(
            model,
            name="keras_model"
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Learning rate: {lr}")
        print(f"Test accuracy: {test_accuracy:.4f}")

print("\nBoth toy runs finished.")
print("Start the MLflow interface with: mlflow ui --backend-store-uri ./mlruns")

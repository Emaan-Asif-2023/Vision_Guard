
import json
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from input_pipeline import create_dataset

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Load datasets
train_ds = create_dataset("train.csv", training=True)
val_ds = create_dataset("val.csv", training=False)

# Build a CNN with tunable hyperparameters
def build_model(hp):
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),

        layers.Conv2D(
            filters=hp.Choice("filters_1", [16, 32]),
            kernel_size=3,
            padding="same",
            activation="relu"
        ),
        layers.MaxPooling2D(),

        layers.Conv2D(
            filters=hp.Choice("filters_2", [32, 64]),
            kernel_size=3,
            padding="same",
            activation="relu"
        ),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            units=hp.Choice("dense_units", [64, 128]),
            activation="relu"
        ),

        layers.Dropout(
            rate=hp.Choice("dropout", [0.3, 0.5])
        ),

        layers.Dense(1, activation="sigmoid")
    ])

    learning_rate = hp.Choice(
        "learning_rate", [0.001, 0.0001]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Configure the random search
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=4,
    executions_per_trial=1,
    directory="keras_tuner_results",
    project_name="visionguard_cnn",
    seed=SEED,
    overwrite=True
)

# Search for the best hyperparameters
tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )
    ]
)

# Retrieve and display the best configuration
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

print("\nBest Hyperparameters")
for name, value in best_hp.values.items():
    print(f"{name}: {value}")


with open("best_hyperparameters.json", "w") as f:
    json.dump(best_hp.values, f, indent=4)


best_model.save("best_tuned_cnn.keras")


tuner.results_summary(num_trials=4)

print("\nSaved best_hyperparameters.json")
print("Saved best_tuned_cnn.keras")

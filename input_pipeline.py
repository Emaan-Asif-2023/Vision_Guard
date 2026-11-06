
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

def load_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, tf.cast(label, tf.float32)

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def create_dataset(csv_file, training=False):
    df = pd.read_csv(csv_file)

    paths = df["filepath"].astype(str).values
    labels = df["label"].astype("float32").values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df), seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        dataset = dataset.map(
            augment_image, num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = create_dataset("train.csv", training=True)
val_ds = create_dataset("val.csv")
test_ds = create_dataset("test.csv")

print("TensorFlow input pipeline created successfully.")
print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())
print("Testing batches:", tf.data.experimental.cardinality(test_ds).numpy())

# Display one augmented training batch.
images, labels = next(iter(train_ds))

plt.figure(figsize=(10, 8))
for i in range(min(8, len(images))):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i].numpy())
    plt.title("AI-generated" if labels[i].numpy() == 1 else "Real")
    plt.axis("off")

plt.tight_layout()
plt.savefig("augmented_batch.png", dpi=150)
print("Saved augmented_batch.png")
plt.show()

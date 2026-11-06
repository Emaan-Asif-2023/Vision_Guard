
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

MODEL_PATH = "best_mobilenetv2_finetuned.keras"
CSV_PATH = "test.csv"
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "Conv_1"

model = tf.keras.models.load_model(MODEL_PATH)
df = pd.read_csv(CSV_PATH)

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image

# Predict on the test set.
images = []
for path in df["filepath"]:
    images.append(load_image(path).numpy())

images = np.array(images, dtype=np.float32)
model_inputs = images * 2.0 - 1.0  # MobileNetV2 preprocessing

probabilities = model.predict(model_inputs, verbose=0).reshape(-1)
predictions = (probabilities >= 0.5).astype(int)
actual = df["label"].to_numpy().astype(int)

correct_indices = np.where(predictions == actual)[0][:4]
wrong_indices = np.where(predictions != actual)[0][:4]

if len(correct_indices) < 4 or len(wrong_indices) < 4:
    raise ValueError("The test set must contain at least 4 correct and 4 incorrect predictions.")


base_model = model.get_layer("mobilenetv2_1.00_224")
grad_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[
        base_model.get_layer(LAST_CONV_LAYER).output,
        base_model.output
    ]
)

def make_gradcam(input_image, predicted_class):
    input_tensor = tf.convert_to_tensor(input_image[None, ...])

    with tf.GradientTape() as tape:
        conv_outputs, output = grad_model(input_tensor, training=False)
        score = output[:, 0]
        if predicted_class == 0:
            score = 1.0 - score

    gradients = tape.gradient(score, conv_outputs)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], IMG_SIZE
    ).numpy().squeeze()

    return heatmap

selected = [
    (idx, "Correct") for idx in correct_indices
] + [
    (idx, "Wrong") for idx in wrong_indices
]

fig, axes = plt.subplots(2, 4, figsize=(16, 9))

for ax, (idx, status) in zip(axes.flat, selected):
    image = images[idx]
    predicted_class = int(predictions[idx])
    true_class = int(actual[idx])

    heatmap = make_gradcam(model_inputs[idx], predicted_class)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8, cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(
        heatmap_color, cv2.COLOR_BGR2RGB
    )

    original_uint8 = np.uint8(np.clip(image * 255, 0, 255))
    overlay = cv2.addWeighted(
        original_uint8, 0.6, heatmap_color, 0.4, 0
    )

    ax.imshow(overlay)
    true_name = "AI-generated" if true_class == 1 else "Real"
    pred_name = "AI-generated" if predicted_class == 1 else "Real"
    ax.set_title(
        f"{status}\nTrue: {true_name}\nPred: {pred_name}",
        fontsize=9
    )
    ax.axis("off")

plt.suptitle("Grad-CAM: 4 Correct and 4 Incorrect Predictions")
plt.tight_layout()
plt.savefig("BONUS_gradcam_results.png", dpi=180, bbox_inches="tight")
plt.show()

print("Saved bonus_results.png")
print("Correct examples:", len(correct_indices))
print("Incorrect examples:", len(wrong_indices))

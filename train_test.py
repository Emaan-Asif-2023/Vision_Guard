
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset"
SEED = 42

records = []

for class_name, label in [("AI", 1), ("Real", 0)]:
    class_dir = os.path.join(DATASET_DIR, class_name)

    for filename in os.listdir(class_dir):
        path = os.path.join(class_dir, filename)

        if os.path.isfile(path):
            records.append({
                "filepath": path.replace("\\", "/"),
                "class_name": class_name,
                "label": label
            })

df = pd.DataFrame(records)

# First, reserve 15% for testing.
train_val, test = train_test_split(
    df,
    test_size=0.15,
    random_state=SEED,
    stratify=df["label"]
)

# Then reserve 15% of the full dataset for validation.
train, val = train_test_split(
    train_val,
    test_size=0.15 / 0.85,
    random_state=SEED,
    stratify=train_val["label"]
)

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

print("Dataset split completed!")
print(f"Training images:   {len(train)}")
print(f"Validation images: {len(val)}")
print(f"Testing images:    {len(test)}")
print("\nClass distribution:")
for name, split in [("Train", train), ("Validation", val), ("Test", test)]:
    print(f"\n{name}:")
    print(split["class_name"].value_counts().to_string())

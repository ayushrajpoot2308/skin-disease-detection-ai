import os
import sys
import json
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_builder import load_or_build

# Config
DATA_DIR   = "dataset"
MODEL_DIR  = "models"
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
VAL_DIR    = os.path.join(DATA_DIR, "val")
VAL_SPLIT  = 0.2
SEED       = 42
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
EPOCHS     = 10

random.seed(SEED)


def create_val_split(train_dir, val_dir, val_split=0.2):
    if os.path.exists(val_dir) and any(os.scandir(val_dir)):
        print(f"[split] '{val_dir}' already exists — skipping split.")
        return
    print(f"[split] Creating {int(val_split*100)}% val split ...")
    os.makedirs(val_dir, exist_ok=True)
    for cls in [d for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d))]:
        src = os.path.join(train_dir, cls)
        dst = os.path.join(val_dir, cls)
        os.makedirs(dst, exist_ok=True)
        imgs = [f for f in os.listdir(src)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        random.shuffle(imgs)
        n = max(1, int(len(imgs) * val_split))
        for fname in imgs[:n]:
            shutil.move(os.path.join(src, fname), os.path.join(dst, fname))
        print(f"  {cls}: {n}/{len(imgs)} moved to val")
    print("[split] Done.\n")


create_val_split(TRAIN_DIR, VAL_DIR, VAL_SPLIT)

train_gen  = ImageDataGenerator(rescale=1./255)
val_gen    = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical", seed=SEED
)
val_data = val_gen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical", seed=SEED
)

NUM_CLASSES = train_data.num_classes
print(f"Classes ({NUM_CLASSES}): {list(train_data.class_indices.keys())}")
print(f"Train: {train_data.samples}  Val: {val_data.samples}\n")

os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
    json.dump(train_data.class_indices, f, indent=2)
print("Class mapping saved.\n")

for model_name in ["CNN", "EfficientNet", "MobileNet", "VGG16", "ResNet50"]:
    print(f"\n{'='*55}\n  Training: {model_name}\n{'='*55}")

    model = load_or_build(model_name, num_classes=NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    save_path = os.path.join(MODEL_DIR, f"{model_name}.keras")  # use .keras format
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                save_path, save_best_only=True,
                monitor="val_accuracy", verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3,
                restore_best_weights=True, verbose=1
            ),
        ]
    )
    print(f"{model_name} saved -> {save_path}")

print("\nAll models trained and saved!")
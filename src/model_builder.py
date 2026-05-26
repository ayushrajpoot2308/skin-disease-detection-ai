
import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, VGG16, ResNet50
from tensorflow.keras import layers, models, regularizers

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def apply_augmentation(x):
    """Inline augmentation — avoids pickling issues with ModelCheckpoint."""
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomContrast(0.15)(x)
    return x


def classification_head(x, num_classes, dropout_rate=0.4, l2=1e-4, name_prefix=""):
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(l2),
                     name=f"{name_prefix}_fc1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop1")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(l2),
                     name=f"{name_prefix}_fc2")(x)
    x = layers.Dropout(dropout_rate / 2, name=f"{name_prefix}_drop2")(x)
    out = layers.Dense(num_classes, activation="softmax",
                       name=f"{name_prefix}_output")(x)
    return out


def build_cnn(input_shape=(224, 224, 3), num_classes=9):
    inputs = layers.Input(shape=input_shape)
    x = apply_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(32, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Block 4
    x = layers.Conv2D(256, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    out = classification_head(x, num_classes, dropout_rate=0.5, name_prefix="cnn")
    return models.Model(inputs, out, name="CNN")


def build_efficientnet(input_shape=(224, 224, 3), num_classes=9):
    inputs = layers.Input(shape=input_shape)
    x = apply_augmentation(inputs)

    base = EfficientNetB0(weights="imagenet", include_top=False,
                          input_shape=input_shape)
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    out = classification_head(x, num_classes, dropout_rate=0.4, name_prefix="effnet")
    return models.Model(inputs, out, name="EfficientNet")


def build_mobilenet(input_shape=(224, 224, 3), num_classes=9):
    inputs = layers.Input(shape=input_shape)
    x = apply_augmentation(inputs)

    base = MobileNetV2(weights="imagenet", include_top=False,
                       alpha=1.4, input_shape=input_shape)
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    out = classification_head(x, num_classes, dropout_rate=0.4, name_prefix="mobilenet")

    model = models.Model(inputs, out, name="MobileNet")

    def unfreeze_top(n_layers=20):
        base.trainable = True
        for layer in base.layers[:-n_layers]:
            layer.trainable = False
        print(f"[MobileNet] Unfrozen top {n_layers} layers.")
    model.unfreeze_top = unfreeze_top
    return model


def build_vgg16(input_shape=(224, 224, 3), num_classes=9):
    inputs = layers.Input(shape=input_shape)
    x = apply_augmentation(inputs)

    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=input_shape)
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    out = classification_head(x, num_classes, dropout_rate=0.5, name_prefix="vgg16")

    model = models.Model(inputs, out, name="VGG16")

    def unfreeze_top_blocks():
        base.trainable = True
        for layer in base.layers:
            if layer.name.startswith(("block1", "block2", "block3")):
                layer.trainable = False
        print("[VGG16] Unfrozen block4 + block5.")
    model.unfreeze_top_blocks = unfreeze_top_blocks
    return model


def build_resnet50(input_shape=(224, 224, 3), num_classes=9):
    inputs = layers.Input(shape=input_shape)
    x = apply_augmentation(inputs)

    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=input_shape)
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    out = classification_head(x, num_classes, dropout_rate=0.4, name_prefix="resnet50")

    model = models.Model(inputs, out, name="ResNet50")

    def unfreeze_top_blocks():
        base.trainable = True
        for layer in base.layers:
            if not (layer.name.startswith("conv4") or layer.name.startswith("conv5")):
                layer.trainable = False
        print("[ResNet50] Unfrozen conv4 + conv5 blocks.")
    model.unfreeze_top_blocks = unfreeze_top_blocks
    return model


BUILDERS = {
    "CNN":          build_cnn,
    "EfficientNet": build_efficientnet,
    "MobileNet":    build_mobilenet,
    "VGG16":        build_vgg16,
    "ResNet50":     build_resnet50,
}


def load_or_build(model_name, input_shape=(224, 224, 3), num_classes=9):
    if model_name not in BUILDERS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(BUILDERS.keys())}")

    # Check .keras first (native), then .h5 (legacy)
    for ext in (".keras", ".h5"):
        path = os.path.join(MODEL_DIR, model_name + ext)
        if os.path.exists(path):
            try:
                print(f"[{model_name}] Loading saved model from {path}")
                return tf.keras.models.load_model(path)
            except Exception as e:
                print(f"[{model_name}] WARNING: Could not load '{path}': {e}")
                print(f"[{model_name}] Deleting corrupted file and rebuilding...")
                os.remove(path)

    print(f"[{model_name}] Building fresh architecture.")
    return BUILDERS[model_name](input_shape=input_shape, num_classes=num_classes)
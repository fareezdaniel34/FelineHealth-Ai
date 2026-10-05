from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation():
    """Random changes applied ONLY during training (switched off automatically
    when predicting), so the model sees slightly different images every epoch."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),       # about +/- 30 degrees
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
            layers.RandomBrightness(0.10, value_range=(0.0, 1.0)),
        ],
        name="augmentation",
    )


def conv_block(x, filters, name):
    """Conv -> BatchNorm -> ReLU -> MaxPool."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    return x


def build_cnn(img_size=128, num_classes=4, filters=(32, 64, 128), dense_units=128,
              dropout=0.5, augment=True):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")   # uint8 0-255
    x = layers.Rescaling(1.0 / 255, name="normalise")(inputs)           # -> [0, 1]
    if augment:
        x = build_augmentation()(x)

    for i, f in enumerate(filters, start=1):
        x = conv_block(x, f, name=f"block{i}")

    # Global average pooling instead of Flatten: far fewer parameters,
    # which reduces overfitting on a small dataset (574 training images).
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name="FelineHealth_CNN")


if __name__ == "__main__":
    build_cnn().summary()
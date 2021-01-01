import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import plot_history

DATA_DIR = "./flower_photos"
NUM_CLASSES = 5
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180
BATCH_SIZE = 32
TRAIN_VAL_SPLIT = 0.2
SEED = 42

AUTOTUNE = tf.data.experimental.AUTOTUNE


def build_image_generator():
    train_ds = keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=TRAIN_VAL_SPLIT,
        subset='training',
        seed=SEED,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE
    ).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    val_ds = keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=TRAIN_VAL_SPLIT,
        subset='validation',
        seed=SEED,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE
    ).cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def build_model():
    augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ])

    model = keras.models.Sequential([
        augmentation,
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES)  # Note: No softmax here
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Note: Set from_logits=True
        metrics=['accuracy']
    )
    return model


def main():
    train_ds, val_ds = build_image_generator()

    model = build_model()
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )
    model.save('saved_model')

    plot_history(history)


if __name__ == '__main__':
    main()

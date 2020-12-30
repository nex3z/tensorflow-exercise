from tensorflow import keras
from tensorflow.keras import layers

import utils

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def main():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = keras.models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_images, train_labels,
        epochs=10,
        validation_data=(test_images, test_labels)
    )
    utils.plot_history(history)

    model.save("./saved_model")


if __name__ == '__main__':
    main()

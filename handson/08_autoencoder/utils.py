import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def reset_session(random_seed=42):
    keras.backend.clear_session()
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def load_fashion_mnist_data():
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train_full = x_train_full.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255
    x_train, x_val = x_train_full[:-5000], x_train_full[-5000:]
    y_train, y_val = y_train_full[:-5000], y_train_full[-5000:]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


def plot_history(history, metrics=None, start=0):
    if metrics is None:
        metrics = [('accuracy', 'val_accuracy'), ('loss', 'val_loss')]
    elif isinstance(metrics, str):
        metrics = [(metrics,)]

    num_metrics = len(metrics)
    plt.figure(figsize=(6 * num_metrics, 4))
    for i, group in enumerate(metrics):
        if isinstance(group, str):
            group = (group,)
        plt.subplot(1, num_metrics, i + 1)
        for key in group:
            epochs = range(len(history.history[key][start:]))
            plt.plot(epochs, history.history[key][start:], label=key)
        plt.legend()
    plt.show()


def reconstruct(model, images):
    reconstructions = model.predict(images)
    num_images = len(images)
    plt.figure(figsize=(num_images * 1.5, 3))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i], cmap='binary')
        plt.axis('off')
        plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(reconstructions[i], cmap='binary')
        plt.axis('off')
    plt.show()


def plot_images(generated_images, num_rows=4):
    num_cols = np.ceil(len(generated_images) / num_rows)
    plt.figure(figsize=(num_cols, num_rows))
    for i in range(len(generated_images)):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(generated_images[i], cmap='binary')
        plt.axis('off')
    plt.show()

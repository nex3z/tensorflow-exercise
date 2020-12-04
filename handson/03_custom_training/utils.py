import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def reset_session(random_seed=42):
    keras.backend.clear_session()
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def load_housing_data(random_state=42):
    housing = fetch_california_housing()

    x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=random_state)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_fashion_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    means = train_images.mean(axis=0, keepdims=True)
    stds = train_images.std(axis=0, keepdims=True)

    # train_images  = train_images / 255.0
    train_images = (train_images - means) / stds

    # test_images = test_images / 255.0
    test_images = (test_images - means) / stds

    return (train_images, train_labels), (test_images, test_labels)


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

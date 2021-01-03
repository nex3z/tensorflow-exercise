import math
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def plot_image(image):
    if len(image.shape) > 3:
        image = np.squeeze(image)
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def plot_images(images, labels=None, class_names=None, one_hot_label=False):
    if isinstance(labels, tf.Tensor):
        labels = labels.numpy()
        
    num_images = len(images)
    num_cols = 8
    num_rows = math.ceil(num_images / num_cols)
    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    for i, image in enumerate(images):
        if len(image.shape) > 3:
            image = np.squeeze(image)
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image)
        if labels is not None:
            label = labels[i]
            if one_hot_label:
                label = np.argmax(label)
            if class_names is not None:
                label = class_names[label]
            plt.title(label)
        plt.axis('off')
    plt.show()


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


import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def plot_images(images, labels, class_names=None, one_hot_label=True):
    num_images = len(images)
    num_cols = 8
    num_rows = math.ceil(num_images / num_cols)
    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image)
        if one_hot_label:
            label = np.argmax(label)
        if class_names is not None:
            label = class_names[label]
        plt.title(label)
        plt.axis('off')
    plt.show()

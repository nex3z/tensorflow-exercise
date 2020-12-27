import os

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

DATA_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv'


def load_data():
    data_path = keras.utils.get_file(os.path.basename(DATA_URL), DATA_URL, cache_dir='./', cache_subdir='')
    df_data = pd.read_csv(data_path, index_col=0)
    time = df_data.index.to_numpy()
    series = df_data['Monthly Mean Total Sunspot Number'].to_numpy()
    return time, series


def plot_series(time, series, start=0, end=None, label=None):
    plt.figure(figsize=(10, 6))
    if not isinstance(series, list):
        series = [series]

    show_label = label is not None
    if label is None:
        label = [None] * len(series)
    elif not isinstance(label, list):
        label = [label]

    for s, l in zip(series, label):
        plt.plot(time[start:end], s[start:end], label=l)

    plt.xlabel("Time")
    plt.ylabel("Value")

    if show_label:
        plt.legend(fontsize=14)
    plt.grid(True)
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


def evaluate(actual, forecast):
    mse = keras.metrics.mean_squared_error(actual, forecast).numpy()
    mae = keras.metrics.mean_absolute_error(actual, forecast).numpy()
    print(f"mse = {mse:.4f}, mae = {mae:.4f}")

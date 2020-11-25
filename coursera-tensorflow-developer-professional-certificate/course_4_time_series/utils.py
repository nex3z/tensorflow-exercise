import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


def trend(time, slope=0.0):
    return slope * time


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def generate_time_series(periods=4 * 365 + 1, baseline=10, slope=0.05, seasonal_periods=365, seasonal_amplitude=40,
                         noise_level=5, noise_seed=42):
    time = np.arange(periods, dtype="float32")

    series = baseline \
        + trend(time, slope) \
        + seasonality(time, period=seasonal_periods, amplitude=seasonal_amplitude) \
        + noise(time, noise_level, seed=noise_seed)

    return series


def evaluate(actual, forecast):
    mse = keras.metrics.mean_squared_error(actual, forecast).numpy()
    mae = keras.metrics.mean_absolute_error(actual, forecast).numpy()
    print(f"mse = {mse:.4f}, mae = {mae:.4f}")


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

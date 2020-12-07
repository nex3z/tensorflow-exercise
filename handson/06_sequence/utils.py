import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def reset_session(random_seed=42):
    keras.backend.clear_session()
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def generate_time_series(batch_size, num_steps, seed=42):
    np.random.seed(seed)
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, num_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))   # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + wave 2
    series += 0.1 * (np.random.rand(batch_size, num_steps) - 0.5)  # + noise
    return series.astype(np.float32)


def load_time_series_data(num_steps=50):
    series = generate_time_series(10000, num_steps + 1)
    x_train, y_train = series[:7000, :num_steps], series[:7000, -1]
    x_val, y_val = series[7000:9000, :num_steps], series[7000:9000, -1]
    x_test, y_test = series[9000:, :num_steps], series[9000:, -1]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_time_series_data_multiple_forecast(history_steps=50, forecast_steps=10, seed=42):
    series = generate_time_series(10000, history_steps + forecast_steps, seed=seed)

    x_train = series[:7000, :history_steps]
    x_val = series[7000:9000, :history_steps]
    x_test = series[9000:, :history_steps]
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)

    y = np.empty((10000, history_steps, forecast_steps))
    for step_ahead in range(1, forecast_steps + 1):
        y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + history_steps]
    y_train = y[:7000]
    y_val = y[7000:9000]
    y_test = y[9000:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def plot_series(data, y=None, y_pred=None):
    plt.figure()
    plt.plot(data, marker='.')
    if y is not None:
        plt.plot(len(data), y, 'bo', markersize=8, label='y')
    if y_pred is not None:
        plt.plot(len(data), y_pred, 'rx', markersize=8, label='y_pred')
    if y is not None or y_pred is not None:
        plt.legend()
    plt.show()


def plot_series_multiple_forecasts(data, y=None, y_pred=None):
    data = np.squeeze(data)
    y = np.squeeze(y)
    y_pred = np.squeeze(y_pred)

    plt.figure()
    plt.plot(data, marker='.')
    if y is not None:
        plt.plot(np.arange(len(data), len(data) + len(y)), y, '-bo', markersize=8, label='y')
    if y_pred is not None:
        plt.plot(np.arange(len(data), len(data) + len(y_pred)), y_pred, '-rx', markersize=8, label='y_pred')
    if y is not None or y_pred is not None:
        plt.legend()
    plt.show()


def plot_history(history, metrics=None, start=0):
    if metrics is None:
        metrics = [('loss', 'val_loss'), ('mae', 'val_mae'), ]
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


def evaluate(y, y_pred):
    mse = keras.metrics.mean_squared_error(y, y_pred).numpy()
    mae = keras.metrics.mean_absolute_error(y, y_pred).numpy()
    print(f"mse = {mse:.8f}, mae = {mae:.8f}")

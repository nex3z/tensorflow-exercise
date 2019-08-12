import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset


def plot_ts(time, series):
    plt.plot(time, series)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def plot_df(df_data, columns=None, figsize=(8, 4)):
    if columns is not None:
        df_data.loc[:, columns].plot(figsize=figsize)
    else:
        df_data.plot(figsize=figsize)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def noise(time, level=1, seed=42):
    rs = np.random.RandomState(seed)
    return rs.randn(len(time)) * level


def trend(time, slope):
    return slope * time


def cos(period):
    t = np.arange(period) / period
    return np.cos(2 * np.pi * t)


def composition(period):
    t = np.arange(period) / period
    return np.where(t < 0.4, np.cos(2 * np.pi * t), 1 / np.exp(3 * t))


def seasonality(time, period, gen=cos, amplitude=1):
    seasonal_data = gen(period)
    return np.resize(seasonal_data * amplitude, len(time))


def load_synthesized_data(split=None):
    time = np.arange(4 * 365 + 1)
    series = 10 + trend(time, slope=0.05) \
        + seasonality(time, period=365, gen=composition, amplitude=40) \
        + noise(time, level=5, seed=42)
    df_data = pd.DataFrame({'y': series}, index=time)
    if split is None:
        return df_data
    else:
        return df_data[:split], df_data[split:]


def build_dataset(series, window_size, batch_size, shuffle_buffer_size):
    dataset = Dataset.from_tensor_slices(series) \
        .window(window_size + 1, shift=1, drop_remainder=True) \
        .flat_map(lambda window: window.batch(window_size + 1)) \
        .shuffle(shuffle_buffer_size).map(lambda window: (window[:-1], window[-1])) \
        .batch(batch_size).prefetch(1)
    return dataset


def validate(df_forecast):
    diff = df_forecast['y'] - df_forecast['y_hat']
    mse = (diff**2).mean()
    mae = diff.abs().mean()
    print("MSE: {}, MAE: {}".format(mse, mae))
    plot_df(df_forecast, columns=['y', 'y_hat'])

    
def validate_model2(model, df_val, window_size):
    df_forecast = df_val.copy()
    df_forecast['y_hat'] = np.nan

    for i, (idx, row) in enumerate(df_forecast.iterrows()):
        if i < window_size:
            continue
        y_val = df_forecast['y'].iloc[(i - window_size):i].values
        forecast_value = model.predict(y_val[np.newaxis])
        df_forecast.loc[idx, 'y_hat'] = forecast_value[0, 0]

    validate(df_forecast[window_size:])


def validate_model(model, df_val, window_size, batch_size=512):
    y_val = df_val['y'].values
    dataset_val = Dataset.from_tensor_slices(y_val) \
        .window(window_size, shift=1, drop_remainder=True) \
        .flat_map(lambda window: window.batch(window_size)) \
        .batch(batch_size).prefetch(1)
    forecast = model.predict(dataset_val)
    df_forecast = df_val[window_size:].copy()
    df_forecast['y_hat'] = forecast[:-1]
    validate(df_forecast)


def plot_history(history, start_epoch=1, metrics=None):
    if metrics is None:
        metrics = ['loss']
    for metric in metrics:
        epoch = range(start_epoch, len(history.history[metric]) + 1)
        metric_value = history.history[metric][(start_epoch - 1):]
        plt.plot(epoch, metric_value, label=metric)
    plt.xlabel("Epoch")
    plt.legend()

    
def clear_env(seed=42):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

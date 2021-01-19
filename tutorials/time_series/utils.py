import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

DATA_FILE = './jena_climate_2009_2016.csv'


def load_data():
    df_data = pd.read_csv(DATA_FILE)
    df_data = df_data[5::6]
    df_data['Date Time'] = pd.to_datetime(df_data['Date Time'], format='%d.%m.%Y %H:%M:%S')

    df_data.loc[df_data['wv (m/s)'] == -9999.0, 'wv (m/s)'] = 0.0
    df_data.loc[df_data['max. wv (m/s)'] == -9999.0, 'max. wv (m/s)'] = 0.0

    # Convert to radians.
    wd_rad = df_data['wd (deg)'] * np.pi / 180

    # Calculate the wind x and y components.
    df_data['Wx'] = df_data['wv (m/s)'] * np.cos(wd_rad)
    df_data['Wy'] = df_data['wv (m/s)'] * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df_data['max Wx'] = df_data['max. wv (m/s)'] * np.cos(wd_rad)
    df_data['max Wy'] = df_data['max. wv (m/s)'] * np.sin(wd_rad)

    # Seasonality
    sec_timestamp = df_data['Date Time'].map(datetime.datetime.timestamp)
    sec_per_day = 24 * 60 * 60
    sec_per_year = 365.2425 * sec_per_day
    df_data['Day sin'] = np.sin(sec_timestamp * (2 * np.pi / sec_per_day))
    df_data['Day cos'] = np.cos(sec_timestamp * (2 * np.pi / sec_per_day))
    df_data['Year sin'] = np.sin(sec_timestamp * (2 * np.pi / sec_per_year))
    df_data['Year cos'] = np.cos(sec_timestamp * (2 * np.pi / sec_per_year))

    # Train test splits
    feature_columns = [
        'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)',
        'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'Wx', 'Wy', 'max Wx', 'max Wy',
        'Day sin', 'Day cos', 'Year sin', 'Year cos'
    ]
    n = len(df_data)
    df_train = df_data[0:int(n * 0.7)][feature_columns]
    df_val = df_data[int(n * 0.7):int(n * 0.9)][feature_columns]
    df_test = df_data[int(n * 0.9):][feature_columns]

    train_mean = df_train.mean()
    train_std = df_train.std()
    df_train = (df_train - train_mean) / train_std
    df_val = (df_val - train_mean) / train_std
    df_test = (df_test - train_mean) / train_std
    return df_train, df_val, df_test


def build_dataset(df_data, window_size, batch_size, label_idx=None, shuffle=True):
    data = np.array(df_data, dtype=np.float32)
    ds = keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=window_size + 1,
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=batch_size,
    )
    if label_idx is not None:
        ds = ds.map(lambda batched_window: (batched_window[:, :-1, :], batched_window[:, -1:, label_idx:label_idx+1]))
    else:
        ds = ds.map(lambda batched_window: (batched_window[:, :-1, :], batched_window[:, -1:, :]))
    return ds


def plot_result(actual, forecast):
    plt.figure(figsize=(8, 4))
    plt.plot(actual, marker='o', label='actual')
    plt.plot(forecast, marker='o', label='forecast')
    plt.legend()
    plt.show()


def evaluate(actual, forecast):
    mse = keras.metrics.mean_squared_error(actual, forecast).numpy()
    mae = keras.metrics.mean_absolute_error(actual, forecast).numpy()
    print(f"mse = {mse:.4f}, mae = {mae:.4f}")

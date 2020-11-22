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

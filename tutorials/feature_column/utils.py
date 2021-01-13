import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

DATA_URL = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'


def load_data():
    data_file = keras.utils.get_file(
        os.path.basename(DATA_URL),
        DATA_URL,
        cache_dir='./',
        cache_subdir='',
        extract=True
    )
    file_path = os.path.join(os.path.splitext(data_file)[0], 'petfinder-mini.csv')

    df_data = pd.read_csv(file_path)
    df_data['target'] = np.where(df_data['AdoptionSpeed'] == 4, 0, 1)
    df_data = df_data.drop(columns=['AdoptionSpeed', 'Description'])

    df_train, df_test = train_test_split(df_data, test_size=0.2)
    df_train, df_val = train_test_split(df_train, test_size=0.2)
    return df_train, df_val, df_test

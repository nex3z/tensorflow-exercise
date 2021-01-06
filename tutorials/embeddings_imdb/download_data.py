import os
import shutil

from tensorflow import keras

DATA_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'


def download_data():
    data_dir = keras.utils.get_file(
        os.path.basename(DATA_URL),
        DATA_URL,
        cache_dir='./',
        cache_subdir='',
        extract=True
    )
    data_dir = os.path.join(os.path.dirname(data_dir), 'aclImdb')

    train_dir = os.path.join(data_dir, 'train')
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

    return data_dir


def main():
    download_data()


if __name__ == '__main__':
    main()

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def reset_session(random_seed=42):
    keras.backend.clear_session()
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def load_text_generation_dataset(num_steps=100, batch_size=32):
    # file_path = keras.utils.get_file(os.path.basename(DATA_URL), DATA_URL)
    # with open(file_path) as f:
    #     text = f.read()
    text = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way— in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only."
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(text)

    window_size = num_steps + 1
    vocab_size = len(tokenizer.word_index)

    encoded = (np.array(tokenizer.texts_to_sequences([text])) - 1)[0]
    dataset = tf.data.Dataset.from_tensor_slices(encoded)
    # dataset = dataset.repeat().window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.map(lambda windows: (windows[:-1], windows[1:]))
    dataset = dataset.shuffle(10000).batch(batch_size)
    dataset = dataset.map(lambda x_batch, y_batch: (tf.one_hot(x_batch, depth=vocab_size), y_batch))
    dataset = dataset.prefetch(1)

    return tokenizer, dataset


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

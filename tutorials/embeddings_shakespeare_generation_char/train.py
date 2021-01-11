from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

DATA_FILE = './shakespeare.txt'
MAX_SEQUENCE_LENGTH = 100

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 10000


def x_y_split(chunk):
    return chunk[:-1], chunk[1:]


def load_data():
    text = open(DATA_FILE, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    train_ds = tf.data.Dataset.from_tensor_slices(text_as_int)\
        .batch(MAX_SEQUENCE_LENGTH + 1, drop_remainder=True) \
        .map(x_y_split) \
        .shuffle(SHUFFLE_BUFFER_SIZE)\
        .batch(BATCH_SIZE, drop_remainder=True)

    return train_ds, char2idx, idx2char


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        layers.Dense(vocab_size)
    ])
    return model


def main():
    train_ds, char2idx, idx2char = load_data()
    np.save('idx2char.npy', idx2char)

    vocab_size = len(char2idx)

    model = build_model(vocab_size, 256, 1024, BATCH_SIZE)

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    history = model.fit(train_ds, epochs=10)

    # model.save('./saved_model')
    model.save_weights('./checkpoints/last_checkpoint')


if __name__ == '__main__':
    main()

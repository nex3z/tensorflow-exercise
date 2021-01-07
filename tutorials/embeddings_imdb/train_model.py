import io
import re
import string

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TRAIN_DIR = 'aclImdb/train'
VAL_SPLIT = 0.2
BATCH_SIZE = 1024
SEED = 42

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE

EMBEDDING_DIM = 16


def load_dataset():
    train_ds = keras.preprocessing.text_dataset_from_directory(
        TRAIN_DIR,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        subset='training',
        seed=SEED
    ).cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = keras.preprocessing.text_dataset_from_directory(
        TRAIN_DIR,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        subset='validation',
        seed=SEED
    ).cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, f'[{re.escape(string.punctuation)}]', '')


def build_model(vectorize_layer):
    model = keras.models.Sequential([
        vectorize_layer,
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name='embedding'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def save_embeddings(vectorize_layer, embedding_layer):
    weights = embedding_layer.get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0: continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


def main():
    train_ds, val_ds = load_dataset()

    vectorize_layer = keras.layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standardization,
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH
    )
    text_ds = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    model = build_model(vectorize_layer)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[tensorboard_callback]
    )

    save_embeddings(vectorize_layer, model.get_layer('embedding'))


if __name__ == '__main__':
    main()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from train import build_model


def generate(model, start, num_generate=1000, temperature=1.0):
    generated = []
    model.reset_states()
    input_eval = tf.expand_dims(start, 0)
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        generated.append(predicted_id)
        input_eval = tf.expand_dims([predicted_id], 0)
    return generated


def main():
    idx2char = np.load('idx2char.npy', allow_pickle=True)
    char2idx = {ch: idx for idx, ch in enumerate(idx2char)}
    vocab_size = len(char2idx)

    def encode(text):
        return [char2idx[c] for c in text]

    def decode(sequence):
        return ''.join([idx2char[i] for i in sequence])

    model = build_model(vocab_size, 256, 1024, 1)
    model.load_weights('./checkpoints/last_checkpoint')
    model.build(tf.TensorShape([1, None]))
    model.summary()

    start = u"ROMEO: "
    generated = generate(model, encode(start), 1000)
    generated = decode(generated)
    print(start + generated)


if __name__ == '__main__':
    main()

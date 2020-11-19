import matplotlib.pyplot as plt
import io

def plot_history(history):
    epochs = range(len(history.history['accuracy']))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='val_accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='loss')
    plt.plot(epochs, history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()


def save_embeddings(tokenizer, layer, vector_file='vecs.tsv', meta_file='meta.tsv'):
    weights = layer.get_weights()[0]
    out_v = io.open(vector_file, 'w', encoding='utf-8')
    out_m = io.open(meta_file, 'w', encoding='utf-8')
    for word_num in range(1, tokenizer.vocab_size):
        word = tokenizer.decode([word_num])
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write("\t".join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()

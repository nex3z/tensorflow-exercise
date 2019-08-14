import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_history(history, start_epoch=1, metrics=None, include_val=True):
    if metrics is None:
        metrics = list(set(map(lambda key: key.replace('val_', ''),  history.history.keys())))

    history_dict = history.history
    epochs = range(start_epoch, len(history_dict[metrics[0]]) + 1)
    plt.figure(figsize=(6 * len(metrics), 4))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, len(metrics), i)
        plt.plot(epochs, history_dict[metric], label=metric)
        if include_val:
            val_metric = 'val_' + metric
            plt.plot(epochs, history_dict[val_metric], label=val_metric)
        plt.legend()
    plt.show()

    
def clear_env(seed=42):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

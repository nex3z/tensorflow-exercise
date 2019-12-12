import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_history(history, start_epoch=1, metrics=None):
    if metrics is None:
        metrics = history.history.keys()
    for metric in metrics:
        epoch = range(start_epoch, len(history.history[metric]) + 1)
        metric_value = history.history[metric][(start_epoch - 1):]
        plt.plot(epoch, metric_value, label=metric)
    plt.xlabel("Epoch")
    plt.legend()

    
def clear_env(seed=42):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

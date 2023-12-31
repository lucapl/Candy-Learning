import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import History


def plot_history(history: History, metrics: list[str] = None):
    """
    Plots the history of neural networks training

    :param history: the history of the training
    :type history: tf.keras.callbacks.History
    :param metrics: list of the metrics to plot. Optional, defaults to None
    :type metrics: list[str]

    :returns: None
    """
    metrics = metrics or ['accuracy', 'loss']

    if 'loss' not in metrics:
        metrics.append('loss')

    for metric in metrics:
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()


def plot_sensors(inputs: list[np.ndarray], targets: list[np.ndarray], n=5):
    """
    Plots the graph of the first n timestamps of the sensors data

    :param inputs: list of the input data
    :type inputs: list[np.ndarray]
    :param targets: list of the output data
    :type targets: list[np.ndarray]
    :param n: number of graphs to plot. Optional, defaults to 5
    :type n: int

    :returns: None
    """
    for i in range(n):
        print(targets[i])
        plt.plot(inputs[i])
        plt.legend([f'sensor {j}' for j in range(3)])
        plt.show()


def plot_sensor_comparison(predictions: list[np.ndarray],
                           inputs: list[np.ndarray],
                           targets: list[np.ndarray],
                           mask: np.ndarray, n=5):
    """
    Compares the autoencoder predictions with the original input for the first n timestamps,
    after applying the mask

    :param predictions: list of the input data predicted by the autoencoder
    :type predictions: list[np.ndarray]
    :param inputs: list of the original input data
    :type inputs: list[np.ndarray]
    :param targets: list of the output data
    :type targets: list[np.ndarray]
    :param mask: mask to apply to the data
    :type mask: np.ndarray
    :param n: number of graphs to plot. Optional, defaults to 5
    :type n: int

    :returns: None
    """
    for i in range(n):
        target = targets[i].astype(bool)

        if not target.any():
            print('no defects')
        else:
            for j in range(5):
                if target[j]:
                    print(f'defect {j + 1}', end=' ')

        legend = [f'predicted sensor {j}' for j in range(3)] + [f'true sensor {j}' for j in range(3)]
        plt.plot(predictions[i][mask[i]])
        plt.plot(inputs[i][mask[i]])
        plt.legend(legend)
        plt.show()

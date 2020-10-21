import logging

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from cost_estimate_api import compute_weights_tf, compute_weights_lstsq


def load_data():
    """
    This code is taken from the tensorflow example on doing regression:
    https://www.tensorflow.org/tutorials/keras/regression
    """
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year']

    raw_dataset = pd.read_csv(url, names=column_names,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)
    dataset = raw_dataset.dropna()
    return dataset



def mpg_example():
    data = load_data()
    x = np.asarray(data[['Cylinders', 'Displacement', 'Weight', 'Horsepower', 'Acceleration']])
    y = np.asarray(data["MPG"])

    weights_tf = compute_weights_tf(x, y)
    weights_lstsq = compute_weights_lstsq(x, y)

    def f_factory(_weights):
        def f(_x):
            result = np.zeros(len(_x)) + _weights[-1]
            for icol in range(len(_weights) - 1):
                result += _x[:, icol]*_weights[icol]
            return result
        return f

    def compute_score(y_target, y_predict):
        return np.mean(np.abs(y_target - y_predict))

    fig, axes = plt.subplots(x.shape[1], 1, figsize=(20, 20))

    f_lstsq = f_factory(weights_lstsq)
    y_lstsq = f_lstsq(x)
    score_lstsq = compute_score(y, y_lstsq)


    f_tf = f_factory(weights_tf)
    y_tf = f_tf(x)
    score_tf = compute_score(y, y_tf)

    print(f"score_tf={score_tf:.4f}, score_lstsq={score_lstsq:.4f}")

    for icol in range(x.shape[1]):
        ax = axes[icol]
        x_plot = x[:, icol]

        ax.scatter(x_plot, y)
        ax.scatter(x_plot, f_lstsq(x), color="green", label="least squares")
        ax.scatter(x_plot, f_tf(x), color="red", label="tensorflow")
        ax.legend()
    # ax.scatter(x_plot, model.predict(x), color="red")

    plt.show()


if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    mpg_example()

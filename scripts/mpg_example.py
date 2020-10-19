import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from cost_estimate_api import cost_estimate_linear_regression
from cost_estimate_api.cost_estimate_linear_regression import lstsq


def load_data():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year']

    raw_dataset = pd.read_csv(url, names=column_names,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)
    dataset = raw_dataset.dropna()
    return dataset


# def build_model(n_features):
#     model = tf.keras.sequential([
#         layers.Dense(units=1, input_shape=[n_features, ])
#     ])
#     model.compile(
#         optimizer=tf.optimizers.Adam(learning_rate=0.001),
#         loss='mean_absolute_error')
#
#     return model
#


def test_cost_estimate_linear_regression_mpg():
    data = load_data()
    # sns.pairplot(data[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
    x = np.asarray(data[['Cylinders', 'Displacement', 'Weight']])
    # x = np.asarray(data["Displacement"]).reshape((-1, 1))
    y = np.asarray(data["MPG"])

    weights_tf = cost_estimate_linear_regression(x, y)
    weights_lstsq = lstsq(x, y)

    def f_factory(_weights):
        def f(_x):
            result = np.zeros(len(_x)) + _weights[-1]
            for icol in range(len(_weights) - 1):
                result += _x[:, icol]*_weights[icol]
            return result
        return f

    fig, ax = plt.subplots(1,1, figsize=(20, 20))

    x_plot = x[:, 0]
    f_lstsq = f_factory(weights_lstsq)
    f_tf = f_factory(weights_tf)

    ax.scatter(x_plot, y)
    ax.scatter(x_plot, f_lstsq(x), color="green", label="least squares")
    ax.scatter(x_plot, f_tf(x), color="red", label="tensorflow")
    # ax.scatter(x_plot, model.predict(x), color="red")

    plt.show()





if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    test_cost_estimate_linear_regression_mpg()

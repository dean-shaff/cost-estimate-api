import unittest
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cost_estimate_api import (
    compute_weights_tf,
    compute_weights_lstsq
)


test_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(test_dir, "test_data")
test_file_path = os.path.join(test_data_dir, "test_data.json")


def load_json_data(file_path):
    with open(file_path, "r") as fd:
        data = json.load(fd)
        data = np.asarray(data["data"])
        x, y = data[:, 1:], data[:, 0]
        return x, y


def load_mpg_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration"]

    raw_dataset = pd.read_csv(url, names=column_names,
                              na_values="?", comment="\t",
                              sep=" ", skipinitialspace=True)
    dataset = raw_dataset.dropna()
    x = np.asarray(dataset[["Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration"]])
    y = np.asarray(dataset[["MPG"]])
    return x, y



class TestLinearRegression(unittest.TestCase):

    def test_compute_weights_mpg(self):
        x, y = load_mpg_data()
        weights_tf = compute_weights_tf(x, y)
        weights_lstsq = compute_weights_lstsq(x, y)

    def test_compute_weights(self):
        x, y = load_json_data(test_file_path)
        weights_tf = compute_weights_tf(x, y)
        weights_lstsq = compute_weights_lstsq(x, y)

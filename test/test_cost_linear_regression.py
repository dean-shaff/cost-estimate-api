import unittest
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cost_estimate_api import cost_estimate_linear_regression


test_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(test_dir, "test_data")
test_file_path = os.path.join(test_data_dir, "test_data.json")


def load_json_data(file_path):
    with open(file_path, "r") as fd:
        return json.load(fd)


class TestCostEstimateLinearRegression(unittest.TestCase):


    def test_cost_estimate_linear_regression(self):
        data = load_json_data(test_file_path)["data"]
        data = np.asarray(data)
        y, x = data[:, 0], data[:, 1:]
        weights = cost_estimate_linear_regression(x, y)
        fig, axes = plt.subplots(2, 2)
        axes[0][0].scatter(x[:, 0], y)
        plt.show()
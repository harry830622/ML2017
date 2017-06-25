#!/usr/bin/env python3

from train import NUM_MODELS

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

import sys

if __name__ == "__main__":
    cleaned_x_test_file_name = sys.argv[1]
    suffix = sys.argv[2]
    output_file_name = sys.argv[3]

    x_test = pd.read_pickle(cleaned_x_test_file_name)
    ids = x_test["id"].values
    x_test = x_test.drop("id", axis=1)

    num_testing_data = x_test.shape[0]
    num_classes = 3

    y_test = np.zeros((NUM_MODELS, num_testing_data, num_classes))
    for i in range(NUM_MODELS):
        model = xgb.Booster(model_file="model_{}_{}".format(suffix, i))
        y_test[i] = model.predict(xgb.DMatrix(x_test))
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        xgb.plot_importance(model, ax=ax, max_num_features=20)
        ax.autoscale()
        plt.tight_layout()
        plt.savefig("feature_importance_{}_{}.png".format(suffix, i))
        plt.close()
    y_test = np.mean(y_test, axis=0)
    y_test = np.argmax(y_test, axis=1)

    status_groups = ["functional", "non functional", "functional needs repair"]
    with open(output_file_name, "w") as output_file:
        output_file.write("id,status_group\n")
        for i, n in zip(ids, y_test):
            output_file.write("{},{}\n".format(int(i), status_groups[n]))

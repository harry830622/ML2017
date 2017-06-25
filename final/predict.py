#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb

import sys
import pickle

if __name__ == "__main__":
    cleaned_x_test_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    x_test = pd.read_pickle(cleaned_x_test_file_name)
    ids = x_test["id"].values
    x_test = x_test.drop("id", axis=1)

    num_testing_data = x_test.shape[0]
    num_classes = 3

    y_test = np.zeros((10, num_testing_data, num_classes))

    for i in range(5):
        model = xgb.Booster(model_file="model_{}_{}".format("xgboost", i))
        y_test[i] = model.predict(xgb.DMatrix(x_test))

    for i in range(5):
        with open("model_{}_{}.p".format("rf", i), "rb") as model_file:
            model = pickle.load(model_file)
        y_test[5 + i] = model.predict_proba(x_test)

    y_test = np.mean(y_test, axis=0)
    y_test = np.argmax(y_test, axis=1)

    status_groups = ["functional", "non functional", "functional needs repair"]
    with open(output_file_name, "w") as output_file:
        output_file.write("id,status_group\n")
        for i, n in zip(ids, y_test):
            output_file.write("{},{}\n".format(int(i), status_groups[n]))

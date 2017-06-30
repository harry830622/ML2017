#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.externals import joblib

import os
import sys

if __name__ == "__main__":
    cleaned_x_test_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    model_dir = sys.argv[3]

    x_test = pd.read_pickle(cleaned_x_test_file_name)
    ids = x_test["id"].values
    x_test = x_test.drop("id", axis=1)

    num_testing_data = x_test.shape[0]
    num_classes = 3

    num_xgb_models = 5
    num_rf_models = 1

    y_test = np.zeros((num_xgb_models + num_rf_models, num_testing_data,
                       num_classes))

    for i in range(num_xgb_models):
        model = xgb.Booster(model_file=os.path.join(model_dir,
                                                    "model_{}_{}".format(
                                                        "xgb", i)))
        y_test[i] = model.predict(xgb.DMatrix(x_test))

    for i in range(num_rf_models):
        model = joblib.load(
            os.path.join(model_dir, "model_{}_{}.p".format("rf", i)))
        y_test[num_xgb_models + i] = model.predict_proba(x_test)

    y_test = np.mean(y_test, axis=0)
    y_test = np.argmax(y_test, axis=1)

    status_groups = ["functional", "non functional", "functional needs repair"]
    with open(output_file_name, "w") as output_file:
        output_file.write("id,status_group\n")
        for i, n in zip(ids, y_test):
            output_file.write("{},{}\n".format(int(i), status_groups[n]))

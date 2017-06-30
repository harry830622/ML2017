#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.externals import joblib

import os
import sys

if __name__ == "__main__":
    cleaned_x_test_file_name = sys.argv[1]
    model_dir = sys.argv[2]

    # Plot feature importance
    x_test = pd.read_pickle(cleaned_x_test_file_name)
    x_test = x_test.drop("id", axis=1)
    model = xgb.Booster(model_file=os.path.join(
        model_dir, "model_{}_{}".format("xgb", 0)))
    y_test = model.predict(xgb.DMatrix(x_test))
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    xgb.plot_importance(model, ax=ax, max_num_features=20)
    ax.autoscale()
    plt.tight_layout()
    plt.savefig("feature_importance_{}_{}.png".format("xgb", 0))
    plt.close()

    model = joblib.load(
        os.path.join(model_dir, "model_{}_{}.p".format("rf", 0)))
    y_test = model.predict_proba(x_test)
    rf_importance = model.feature_importances_
    indices = np.argsort(rf_importance)[::-1][:20]
    plt.figure()
    plt.title("Feature Importance")
    plt.barh(
        range(rf_importance[indices].shape[0]), rf_importance[indices][::-1])
    plt.yticks(
        range(rf_importance[indices].shape[0]),
        x_test.columns.values[indices][::-1])
    plt.tight_layout()
    plt.savefig("feature_importance_{}_{}.png".format("rf", 0))
    plt.close()

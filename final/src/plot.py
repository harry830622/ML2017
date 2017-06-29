#!/usr/bin/env python3

import xgboost as xgb
import matplotlib.pyplot as plt

import os

if __name__ == "__main__":
    cleaned_x_test_file_name = sys.argv[1]
    model_dir = sys.argv[2]

    # Plot feature importance
    x_test = x_test.drop("id", axis=1)
    model = xgb.Booster(model_file=os.path.join(
        model_dir, "model_{}_{}".format("xgb", i)))
    y_test = model.predict(xgb.DMatrix(x_test))
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    xgb.plot_importance(model, ax=ax, max_num_features=20)
    ax.autoscale()
    plt.tight_layout()
    plt.savefig("feature_importance_{}_{}.png".format("xgb", 0))
    plt.close()

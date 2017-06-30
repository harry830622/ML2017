#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.externals import joblib

import os
import sys

if __name__ == "__main__":
    x_train_file_name = sys.argv[1]
    y_train_file_name = sys.argv[2]
    x_test_file_name = sys.argv[3]
    cleaned_x_train_file_name = sys.argv[4]
    cleaned_x_test_file_name = sys.argv[5]
    model_dir = sys.argv[6]

    x_train = pd.read_pickle(cleaned_x_train_file_name)
    y_train = pd.read_csv(y_train_file_name)
    x_test = pd.read_pickle(cleaned_x_test_file_name)

    # Plot feature importance
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

    x_train = pd.read_csv(x_train_file_name, parse_dates=["date_recorded"])
    y_train = pd.read_csv(y_train_file_name)
    x_test = pd.read_csv(x_test_file_name, parse_dates=["date_recorded"])
    x_train_test = x_train.append(x_test, ignore_index=True)

    # Observe if "population_below" is truly effective
    print(y_train[x_train["population_below"] == 1]["status_group"]
          .value_counts())

    # Missing value

    na_values = [
        0,
        -2e-8,
        "",
        "None",
        "none",
        "Unknown",
        "unknown",
        "Not Known",
        "not known",
    ]

    missing_value_ratio = {}
    for k, v in x_train_test.iteritems():
        if k == "id" or v.dtype == "datetime64[ns]":
            continue

        if v.dtype == "object":
            v = v.fillna("None")
        else:
            v = v.fillna(0)

        num_missing_values = 0 if True not in v.isin(na_values).value_counts(
        ).index else v.isin(na_values).value_counts()[True]
        missing_value_ratio[k] = num_missing_values / x_train_test.shape[0]

    missing_value_ratio = pd.Series(
        missing_value_ratio, name="missing_value_ratio")
    missing_value_ratio = missing_value_ratio.sort_values(ascending=False)
    missing_value_ratio = missing_value_ratio.iloc[:20].iloc[::-1]

    plt.figure()
    plt.title("Missing Value Ratio")
    missing_value_ratio.plot(kind="barh")
    plt.tight_layout()
    plt.savefig("missing_value_ratio.png")
    plt.close()

    # # of unique values

    unique_value = {}
    for k, v in x_train_test.iteritems():
        if v.dtype == "object":
            unique_value[k] = v.value_counts().shape[0]
    unique_value = pd.Series(unique_value, name="unique_value")
    unique_value = unique_value.sort_values(ascending=False)
    unique_value = unique_value.iloc[:20].iloc[::-1]

    plt.figure()
    plt.title("# of Unique Values")
    unique_value.plot(kind="barh")
    plt.tight_layout()
    plt.savefig("unique_value.png")
    plt.close()

    # Observe the relevance between every feature and status
    features = ["quantity"]
    xy_train = x_train.join(y_train["status_group"])
    for feature in features:
        status = xy_train.groupby(feature)["status_group"]
        ratio = pd.DataFrame(status.value_counts() / status.count())
        print(ratio.sort_values("status_group"))
        break
        ratio["source"] = ratio.index.get_level_values(feature)
        ratio.columns = ["n", "status", feature]
        ratio = ratio.pivot(index=feature, columns="status", values="n")
        ratio["tot"] = ratio.sum(axis=1)
        ratio.sort_values(
            by="tot", ascending=True)[[
                "functional", "functional needs repair", "non functional"
            ]].plot(
                kind="barh",
                stacked="True",
                color=sns.color_palette(),
                fontsize=15,
                legend=False)
        plt.legend(loc="upper right", bbox_to_anchor=(1.9, 0.5), fontsize=15)
        plt.ylabel(feature, fontsize=23)

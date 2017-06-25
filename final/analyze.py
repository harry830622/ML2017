#!/usr/bin/env python3

import numpy as np
import pandas as pd

import sys
import math

if __name__ == "__main__":
    x_train_file_name = sys.argv[1]
    x_test_file_name = sys.argv[2]

    x_train = pd.read_csv(x_train_file_name, parse_dates=["date_recorded"])
    x_test = pd.read_csv(x_test_file_name, parse_dates=["date_recorded"])

    x_train_test = x_train.append(x_test, ignore_index=True)

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

    missing_ratio = {}
    for k, v in x_train_test.iteritems():
        if k == "id" or v.dtype == "datetime64[ns]":
            continue

        if v.dtype == "object":
            v = v.fillna("None")
        else:
            v = v.fillna(0)

        num_missing_values = 0 if True not in v.isin(na_values).value_counts(
        ).index else v.isin(na_values).value_counts()[True]
        missing_ratio[k] = num_missing_values / x_train_test.shape[0]

    missing_ratio = pd.Series(missing_ratio, name="missing_ratio")
    missing_ratio = missing_ratio.sort_values(ascending=False)
    print(missing_ratio)

    # # of unique values

    unique_value = {}
    for k, v in x_train_test.iteritems():
        if v.dtype == "object":
            unique_value[k] = v.value_counts().shape[0]
    unique_value = pd.Series(unique_value, name="unique_value")
    unique_value = unique_value.sort_values(ascending=False)
    print(unique_value)

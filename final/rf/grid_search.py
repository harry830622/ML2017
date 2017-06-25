#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import sys


def search(x_train, y_train):
    for i in [2, 5, 10, 20]:
        for j in [3, 5, 10, 20]:
            model = RandomForestClassifier(
                n_estimators=1000,
                min_samples_split=i,
                max_features=j,
                n_jobs=-1,
                verbose=1)
            np.save("cv_sample_{}_feature_{}.npy".format(i, j),
                    cross_val_score(
                        model, x_train, y_train, cv=5, n_jobs=-1, verbose=1))


if __name__ == "__main__":
    cleaned_x_train_file_name = sys.argv[1]
    y_train_file_name = sys.argv[2]

    x_train = pd.read_pickle(cleaned_x_train_file_name)
    y_train = pd.read_csv(y_train_file_name)
    x_train = x_train.drop("id", axis=1)
    y_train = y_train["status_group"].map({
        "functional": 0,
        "non functional": 1,
        "functional needs repair": 2,
    })

    search(x_train, y_train)

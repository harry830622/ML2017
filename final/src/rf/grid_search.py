#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import os
import sys


def search(x_train, y_train):
    sample = [2, 5, 10, 20]
    feature = [10, 20, 30, 50, "auto"]

    score = {}
    for i in sample:
        for j in feature:
            model = RandomForestClassifier(
                n_estimators=1000,
                min_samples_split=i,
                max_features=j,
                n_jobs=-1,
                verbose=1)
            score["sample_{}_feature_{}".format(i, j)] = cross_val_score(
                model, x_train, y_train, cv=5, n_jobs=-1, verbose=1)

    return score


if __name__ == "__main__":
    cleaned_x_train_file_name = sys.argv[1]
    y_train_file_name = sys.argv[2]
    save_dir = sys.argv[3]

    x_train = pd.read_pickle(cleaned_x_train_file_name)
    y_train = pd.read_csv(y_train_file_name)
    x_train = x_train.drop("id", axis=1)
    y_train = y_train["status_group"].map({
        "functional": 0,
        "non functional": 1,
        "functional needs repair": 2,
    })

    score = search(x_train, y_train)

    for k, v in score.items():
        np.save(
            os.path.join(save_dir, "cv_score_{}_{}.npy".format("rf", k)), v)

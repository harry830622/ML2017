#!/usr/bin/env python3

import pandas as pd
import xgboost as xgb

import sys


def search(x_train, y_train):
    depth = [n for n in range(4, 19, 2)]
    eta = [n / 100 for n in range(2, 6)]
    subsample = [n / 10 for n in range(6, 10)]
    colsample = [n / 10 for n in range(6, 10)]

    for d in depth:
        for e in eta:
            for s in subsample:
                for c in colsample:
                    history = xgb.cv(
                        {
                            "max_depth": d,
                            "eta": e,
                            "subsample": s,
                            "colsample_bytree": c,
                            "num_class": 3,
                            "objective": "multi:softprob",
                        },
                        xgb.DMatrix(x_train, label=y_train),
                        num_boost_round=1000,
                        nfold=5,
                        early_stopping_rounds=30)

                    history.to_pickle(
                        "cv_depth_{}_eta_{}_subsample_{}_colsample_{}.p".
                        format(d, e, s, c))


if __name__ == "__main__":
    cleaned_x_train_file_name = sys.argv[1]
    y_train_file_name = sys.argv[2]

    x_train = pd.read_pickle(cleaned_x_train_file_name)
    y_train = pd.read_csv(y_train_file_name)

    y_train = y_train["status_group"].map({
        "functional": 0,
        "non functional": 1,
        "functional needs repair": 2,
    })

    search(x_train, y_train)

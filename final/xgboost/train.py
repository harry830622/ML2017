#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb

import os
import sys
import random
import pickle

NUM_MODELS = 10

if __name__ == "__main__":
    cleaned_x_train_file_name = sys.argv[1]
    y_train_file_name = sys.argv[2]
    cv_dir = sys.argv[3]
    suffix = sys.argv[4]

    x_train = pd.read_pickle(cleaned_x_train_file_name)
    y_train = pd.read_csv(y_train_file_name)
    x_train = x_train.drop("id", axis=1)
    y_train = y_train["status_group"].map({
        "functional": 0,
        "non functional": 1,
        "functional needs repair": 2,
    })

    cv_result = pd.DataFrame(columns=[
        "error", "best_num_rounds", "depth", "eta", "subsample", "colsample"
    ])
    with os.scandir(cv_dir) as it:
        for e in it:
            cv = pd.read_pickle(e.path)
            param = {
                k: float(v)
                for k, v in zip(cv_result.columns[2:], [
                    s for i, s in enumerate(
                        e.name.rpartition(".p")[0].split("_"))
                    if i > 0 and i % 2 == 0
                ])
            }
            cv_error = cv["test-merror-mean"][-1:]
            row = {
                "error": cv_error.values[0],
                "best_num_rounds": cv_error.index[0],
            }
            row.update(param)
            cv_result = cv_result.append(row, ignore_index=True)
    cv_result.to_csv("cv_result.csv")

    SEED = 19940622
    random.seed(SEED)

    for i, (k, v) in enumerate(
            cv_result.nsmallest(NUM_MODELS, "error").iterrows()):
        best_param = v.to_dict()
        print(best_param)
        model = xgb.train(
            {
                "max_depth": int(best_param["depth"]),
                "eta": best_param["eta"],
                "subsample": best_param["subsample"],
                "colsample_bytree": best_param["colsample"],
                "num_class": 3,
                "objective": "multi:softprob",
                "seed": int(random.random() * 1e8),
            },
            xgb.DMatrix(x_train, label=y_train),
            # num_boost_round=int(best_param["best_num_rounds"]))
            num_boost_round=1000)
        model.save_model("model_{}_{}".format(suffix, i))

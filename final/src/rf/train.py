#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

import os
import sys

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

    cv_result = pd.DataFrame(columns=["score", "sample", "feature"])
    with os.scandir(cv_dir) as it:
        for e in it:
            score = np.load(e.path)
            param = {
                k: v
                for k, v in zip(cv_result.columns[1:], [
                    s for i, s in enumerate(
                        e.name.rpartition(".npy")[0].split("_"))
                    if i > 0 and i % 2 == 0
                ])
            }
            row = {"score": np.mean(score)}
            row.update(param)
            cv_result = cv_result.append(row, ignore_index=True)
    cv_result.to_csv("cv_result.csv")

    SEED = 19940622
    NUM_MODELS = 10

    for i, (k, v
            ) in enumerate(cv_result.nlargest(NUM_MODELS, "score").iterrows()):
        best_param = v.to_dict()
        print(best_param)
        model = RandomForestClassifier(
            n_estimators=1000,
            min_samples_split=int(best_param["sample"]),
            max_features=int(best_param["feature"])
            if best_param["feature"] != "auto" else "auto",
            bootstrap=True,
            oob_score=True,
            random_state=SEED,
            n_jobs=-1,
            verbose=1)
        model.fit(x_train, y_train)
        print(i, model.score(x_train, y_train), model.oob_score_)
        joblib.dump(model, "model_{}_{}.p".format(suffix, i))

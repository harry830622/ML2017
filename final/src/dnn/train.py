#!/usr/bin/env python3

import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

NUM_MODELS = 1

if __name__ == "__main__":
    cleaned_x_train_file_name = sys.argv[1]
    y_train_file_name = sys.argv[2]
    suffix = sys.argv[3]

    x_train = pd.read_pickle(cleaned_x_train_file_name)
    y_train = pd.read_csv(y_train_file_name)

    y_train = y_train["status_group"].map({
        "functional": 0,
        "non functional": 1,
        "functional needs repair": 2,
    })

    print(x_train.values)

    features = Input(shape=(x_train.shape[1], ), name="Features")

    x = Dense(64, activation="relu", name="Dense-1")(x)
    x = Dense(128, activation="relu", name="Dense-2")(x)
    x = Dense(256, activation="relu", name="Dense-3")(x)

    output = Dense(3, activation="softmax", name="StatusGroup")(x)

    model = Model(features, output)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

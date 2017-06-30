#!/usr/bin/env python3

import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.utils import to_categorical

import sys
import pickle

if __name__ == "__main__":
    cleaned_x_train_file_name = sys.argv[1]
    y_train_file_name = sys.argv[2]
    suffix = sys.argv[3]

    x_train = pd.read_pickle(cleaned_x_train_file_name)
    y_train = pd.read_csv(y_train_file_name)
    x_train = x_train.drop("id", axis=1)
    y_train = y_train["status_group"].map({
        "functional": 0,
        "non functional": 1,
        "functional needs repair": 2,
    })

    num_classes = 3

    x_train = x_train.values
    y_train = y_train.values
    y_train = to_categorical(y_train, num_classes=num_classes)

    features = Input(shape=(x_train.shape[1], ), name="Features")

    x = features
    x = Dense(64, activation="relu", name="Dense-1")(x)
    x = Dense(128, activation="relu", name="Dense-2")(x)
    x = Dense(256, activation="relu", name="Dense-3")(x)
    x = Dense(512, activation="relu", name="Dense-4")(x)

    output = Dense(num_classes, activation="softmax", name="StatusGroup")(x)

    model = Model(features, output)

    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])

    model.summary()

    history = model.fit(
        x_train, y_train, batch_size=128, epochs=50, validation_split=0.1)

    model.save("model_{}_{}.h5".format(suffix, 0))

    with open("history_{}_{}.p".format(suffix, 0), "wb") as history_file:
        pickle.dump({"history": history.history}, history_file)

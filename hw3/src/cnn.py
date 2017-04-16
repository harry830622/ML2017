#!/usr/bin/env python

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout

import numpy as np
import pickle
import sys
import os

model_file_name = "./model.p"
is_model_existed = os.path.isfile(model_file_name)

x_test_file_name = "./data/x_test.p"
x_test = []
with open(x_test_file_name, "rb") as x_test_file:
    x_test = pickle.load(x_test_file)
x_test = np.array(x_test, dtype=np.float64).reshape((len(x_test), 48, 48, 1))
x_test /= 255

if not is_model_existed:
    x_train_file_name = "./data/x_train.p"
    x_train = []
    with open(x_train_file_name, "rb") as x_train_file:
        x_train = pickle.load(x_train_file)

    y_train_file_name = "./data/y_train.p"
    y_train = []
    with open(y_train_file_name, "rb") as y_train_file:
        y_train = pickle.load(y_train_file)

    x_train = np.array(
        x_train, dtype=np.float64).reshape((len(x_train), 48, 48, 1))
    y_train = np.array(y_train, dtype=np.float64)

    x_train /= 255

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation("relu"))

    model.add(Dropout(0.5))

    model.add(Dense(7))
    model.add(Activation("softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    model.fit(
        x_train, y_train, batch_size=128, epochs=12, validation_split=0.1)

    m = {}
    m["config"] = model.to_json()
    m["weights"] = model.get_weights()
    with open(model_file_name, "wb") as model_file:
        pickle.dump(m, model_file)
else:
    with open(model_file_name, "rb") as model_file:
        m = pickle.load(model_file)
        model = model_from_json(m["config"])
        model.set_weights(m["weights"])

model.summary()

y_test = model.predict_classes(x_test)

output_file_name = sys.argv[2]
with open(output_file_name, "w") as output_file:
    output_file.write("id,label\n")
    for i, n in enumerate(y_test):
        output_file.write("%d,%d\n" % (i, n))

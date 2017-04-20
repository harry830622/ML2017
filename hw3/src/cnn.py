#!/usr/bin/env python

import numpy as np
import pickle
import sys
import os

from keras import backend as K
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.utils import to_categorical, plot_model

testing_x_file_name = sys.argv[1]
model_file_name = sys.argv[2]

testing_x = []
with open(testing_x_file_name, "rb") as testing_x_file:
    testing_x = pickle.load(testing_x_file)
testing_x = np.array(
    testing_x, dtype=np.float64).reshape((len(testing_x), 48, 48, 1))

testing_x /= 255

is_model_existed = os.path.isfile(model_file_name)

if not is_model_existed:
    training_x_file_name = sys.argv[3]
    training_y_file_name = sys.argv[4]

    training_x = []
    with open(training_x_file_name, "rb") as training_x_file:
        training_x = pickle.load(training_x_file)

    training_y = []
    with open(training_y_file_name, "rb") as training_y_file:
        training_y = pickle.load(training_y_file)
    to_categorical(training_y, num_classes=7)

    training_x = np.array(
        training_x, dtype=np.float64).reshape((len(training_x), 48, 48, 1))
    training_y = np.array(training_y, dtype=np.float64)

    training_x /= 255

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(48, 48, 1)))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (5, 5)))
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
        training_x,
        training_y,
        batch_size=128,
        epochs=20,
        validation_split=0.1,
        callbacks=[TensorBoard()])

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

# plot_model(model, to_file="model.png")

testing_y = model.predict_classes(testing_x)

output_file_name = sys.argv[2]
with open(output_file_name, "w") as output_file:
    output_file.write("id,label\n")
    for i, n in enumerate(testing_y):
        output_file.write("%d,%d\n" % (i, n))

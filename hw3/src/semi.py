#!/usr/bin/env python

import numpy as np
import pickle
import sys

from keras import backend as K
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.utils import to_categorical, plot_model

training_x_file_name = sys.argv[1]
training_y_file_name = sys.argv[2]
testing_x_file_name = sys.argv[3]
model_file_name = sys.argv[4]

num_classes = 7

training_x = []
with open(training_x_file_name, "rb") as training_x_file:
    training_x = pickle.load(training_x_file)

training_y = []
with open(training_y_file_name, "rb") as training_y_file:
    training_y = pickle.load(training_y_file)
training_y = to_categorical(training_y, num_classes=num_classes)

testing_x = []
with open(testing_x_file_name, "rb") as testing_x_file:
    testing_x = pickle.load(testing_x_file)

training_x = np.array(
    training_x, dtype=np.float64).reshape((len(training_x), 48, 48, 1))
training_y = np.array(training_y, dtype=np.float64)
testing_x = np.array(
    testing_x, dtype=np.float64).reshape((len(testing_x), 48, 48, 1))

training_x /= 255
testing_x /= 255

num_testing_xs = testing_x.shape[0]

i = 0
while i < num_testing_xs:
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(32, (5, 5)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation("relu"))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation("relu"))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    model.summary()

    # plot_model(model, to_file="model.png")

    model.fit(
        training_x,
        training_y,
        batch_size=128,
        epochs=20,
        validation_split=0.1,
        callbacks=[TensorBoard()])

    testing_y = model.predict_classes(testing_x)
    testing_y = to_categorical(testing_y, num_classes=num_classes)

    if i + 1000 <= num_testing_xs:
        training_x = np.concatenate((training_x, testing_x[i:i + 1000]))
        training_y = np.concatenate((training_y, testing_y[i:i + 1000]))
    else:
        training_x = np.concatenate((training_x, testing_x))
        training_y = np.concatenate((training_y, testing_y))
    i += 1000

with open(model_file_name, "wb") as model_file:
    pickle.dump({
        "config": model.to_json(),
        "weights": model.get_weights()
    }, model_file)

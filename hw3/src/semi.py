#!/usr/bin/env python3

import numpy as np
import pickle
import sys

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import to_categorical

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

shuffle_index = np.arange(len(training_x))
np.random.seed(19940622)
np.random.shuffle(shuffle_index)
training_x = training_x[shuffle_index]
training_y = training_y[shuffle_index]

num_validating_x = training_x.shape[0] // 10
validating_x = training_x[:num_validating_x]
training_x = training_x[num_validating_x:]
validating_y = training_y[:num_validating_x]
training_y = training_y[num_validating_x:]

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True)

num_testing_xs = testing_x.shape[0]

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1)))
model.add(Activation("relu"))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation("softmax"))

model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

history = {"acc": [], "val_acc": [], "loss": [], "val_loss": []}

batch_size = 256

tmp_history = model.fit_generator(
    train_datagen.flow(training_x, training_y, batch_size=batch_size),
    steps_per_epoch=1000,
    validation_data=(validating_x, validating_y),
    epochs=50,
    callbacks=[EarlyStopping(monitor="val_loss", patience=2), TensorBoard()])

history["acc"] += tmp_history.history["acc"]
history["val_acc"] += tmp_history.history["val_acc"]
history["loss"] += tmp_history.history["loss"]
history["val_loss"] += tmp_history.history["val_loss"]

num_new_training_xs = 0
nth_iteration = 0
while num_new_training_xs < num_testing_xs * 0.7:
    nth_iteration += 1

    predicted_y = model.predict(testing_x)

    thresh = 0.8
    for i, ys in enumerate(predicted_y):
        predicted_classes = list(map(lambda y: y >= thresh, ys))
        if any(predicted_classes):
            training_x = np.concatenate((training_x, [testing_x[i]]))
            training_y = np.concatenate(
                (training_y,
                 [list(map(lambda y: 1 if y else 0, predicted_classes))]))
            num_new_training_xs += 1

    print("{:d} nth self-training, {:d} test labels".format(
        nth_iteration, num_new_training_xs))

    tmp_history = model.fit_generator(
        train_datagen.flow(training_x, training_y, batch_size=batch_size),
        steps_per_epoch=1000,
        validation_data=(validating_x, validating_y),
        epochs=50,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=2), TensorBoard()
        ])

    history["acc"] += tmp_history.history["acc"]
    history["val_acc"] += tmp_history.history["val_acc"]
    history["loss"] += tmp_history.history["loss"]
    history["val_loss"] += tmp_history.history["val_loss"]

model.save(model_file_name)

if len(sys.argv) > 5:
    dump_file_name = sys.argv[5]
    with open(dump_file_name, "wb") as dump_file:
        pickle.dump({"history": history}, dump_file)

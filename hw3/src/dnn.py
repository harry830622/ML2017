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
model_file_name = sys.argv[3]

num_classes = 7

training_x = []
with open(training_x_file_name, "rb") as training_x_file:
    training_x = pickle.load(training_x_file)

training_y = []
with open(training_y_file_name, "rb") as training_y_file:
    training_y = pickle.load(training_y_file)
training_y = to_categorical(training_y, num_classes=num_classes)

training_x = np.array(
    training_x, dtype=np.float64).reshape((len(training_x), 48, 48, 1))
training_y = np.array(training_y, dtype=np.float64)

training_x /= 255

num_validating_x = training_x.shape[0] // 10
validating_x = training_x[:num_validating_x]
training_x = training_x[num_validating_x:]
validating_y = training_y[:num_validating_x]
training_y = training_y[num_validating_x:]

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

model = Sequential()

model.add(Flatten(input_shape=(48, 48, 1)))

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

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

batch_size = 256
history = model.fit_generator(
    train_datagen.flow(training_x, training_y, batch_size=batch_size),
    steps_per_epoch=1000,
    validation_data=(validating_x, validating_y),
    epochs=50,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3), TensorBoard()])

model.save(model_file_name)

if sys.argv.length > 4:
    dump_file_name = sys.argv[4]
    with open(dump_file_name, "wb") as dump_file:
        pickle.dump({"history": history["history"]}, dump_file)

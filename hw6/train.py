#!/usr/bin/env python3

import mf

from extract import extract_training_xy

import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
import pickle

SEED = 19940622
VALIDATION_RATIO = 0.2

if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # set_session(tf.Session(config=config))

    np.random.seed(SEED)

    training_file_name = sys.argv[1]
    suffix = sys.argv[2]

    model_file_name = "model_{}.h5".format(suffix)
    history_file_name = "history_{}.p".format(suffix)

    training_x, training_y = extract_training_xy(training_file_name)
    training_x = np.array(training_x)
    training_y = np.array(training_y)

    model = mf.build()
    model.summary()

    num_training_x = training_x.shape[0]
    indices = np.arange(num_training_x)
    np.random.shuffle(indices)
    training_x = training_x[indices]
    training_y = training_y[indices]
    num_validating_x = int(num_training_x * VALIDATION_RATIO)
    splited_training_x = training_x[:-num_validating_x]
    splited_training_y = training_y[:-num_validating_x]
    splited_validating_x = training_x[-num_validating_x:]
    splited_validating_y = training_y[-num_validating_x:]
    history = model.fit(
        np.hsplit(splited_training_x, 2),
        splited_training_y,
        validation_data=(np.hsplit(splited_validating_x, 2),
                         splited_validating_y),
        batch_size=512,
        epochs=100,
        callbacks=[
            EarlyStopping(
                monitor="val_mean_squared_error", patience=5, verbose=1),
            ModelCheckpoint(
                model_file_name,
                save_best_only=True,
                save_weights_only=True,
                monitor="val_mean_squared_error",
                verbose=1),
        ])

    model.load_weights(model_file_name)
    validating_rmse = np.sqrt(
        model.evaluate(
            np.hsplit(splited_validating_x, 2), splited_validating_y)[1])
    print("\n\n\033[1m\033[92mRMSE on validation set: {:f}\033[0m\n".format(
        validating_rmse))

    with open(history_file_name, "wb") as history_file:
        pickle.dump(history.history, history_file)

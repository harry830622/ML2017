#!/usr/bin/env python3

import mf

from config import IS_NORMALIZED
from extract import extract_xy_train

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
    y_mean_file_name = "y_mean_{}.p".format(suffix)

    x_train, y_train, y_mean, y_mean_index_by_movie_id = extract_xy_train(
        training_file_name, is_normalized=IS_NORMALIZED)

    model = mf.build()
    model.summary()

    num_x_train = x_train.shape[0]
    indices = np.arange(num_x_train)
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    num_x_validation = int(num_x_train * VALIDATION_RATIO)
    splited_x_train = x_train[:-num_x_validation]
    splited_y_train = y_train[:-num_x_validation]
    splited_x_validation = x_train[-num_x_validation:]
    splited_y_validation = y_train[-num_x_validation:]
    history = model.fit(
        np.hsplit(splited_x_train, 2),
        splited_y_train,
        validation_data=(np.hsplit(splited_x_validation, 2),
                         splited_y_validation),
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
            np.hsplit(splited_x_validation, 2), splited_y_validation)[1])
    print("\n\n\033[1m\033[32mRMSE on validation set: {:f}\033[0m\n".format(
        validating_rmse))

    with open(history_file_name, "wb") as history_file:
        pickle.dump(history.history, history_file)

    with open(y_mean_file_name, "wb") as y_mean_file:
        pickle.dump({
            "y_mean": y_mean,
            "y_mean_index_by_movie_id": y_mean_index_by_movie_id,
        }, y_mean_file)

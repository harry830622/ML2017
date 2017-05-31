#!/usr/bin/env python3

import mf
import dnn

from config import METHOD
from config import SEED, NUM_MODELS, VALIDATION_RATIO
from config import NUM_USERS, NUM_MOVIES, LATENT_DIMENSION
from config import IS_NORMALIZED, IS_REGULARIZED, LAMBDA, IS_BIASED
from extract import extract_xy_train, extract_users, extract_movies

import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import sys
import pickle

if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # set_session(tf.Session(config=config))

    np.random.seed(SEED)

    pwd = sys.argv[1]
    suffix = sys.argv[2]

    training_file_name = os.path.join(pwd, "train.csv")
    users_file_name = os.path.join(pwd, "users.csv")
    movies_file_name = os.path.join(pwd, "movies.csv")

    if METHOD == "DNN":
        IS_BIASED = False

    x_train, y_train = extract_xy_train(
        training_file_name, is_normalized=IS_NORMALIZED, is_biased=IS_BIASED)
    users = extract_users(users_file_name, num_users=NUM_USERS)
    movies = extract_movies(movies_file_name, num_movies=NUM_MOVIES)

    num_x_train = x_train.shape[0]
    indices = np.arange(num_x_train)
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    num_x_valid = int(num_x_train * VALIDATION_RATIO)
    splited_x_train = x_train[:-num_x_valid]
    splited_y_train = y_train[:-num_x_valid]
    splited_x_valid = x_train[-num_x_valid:]
    splited_y_valid = y_train[-num_x_valid:]

    num_splited_x_train = splited_x_train.shape[0]
    indices = np.arange(num_splited_x_train)
    for i in range(NUM_MODELS):
        model_file_name = os.path.join(pwd, "model_{}_{:d}.h5".format(
            suffix, i))
        history_file_name = os.path.join(pwd, "history_{}_{:d}.p".format(
            suffix, i))

        if METHOD == "MF":
            model = mf.build(
                num_users=NUM_USERS,
                num_movies=NUM_MOVIES,
                latent_dimension=LATENT_DIMENSION,
                is_regularized=IS_REGULARIZED,
                lamda=LAMBDA,
                is_biased=IS_BIASED)
        if METHOD == "DNN":
            model = dnn.build(users, movies)

        model.summary()

        np.random.shuffle(indices)
        splited_x_train = splited_x_train[indices]
        splited_y_train = splited_y_train[indices]
        index = int(num_splited_x_train * 0.8)

        history = model.fit(
            np.hsplit(splited_x_train[:index], x_train.shape[1]),
            splited_y_train[:index],
            validation_data=(np.hsplit(splited_x_train[index:],
                                       x_train.shape[1]),
                             splited_y_train[index:]),
            batch_size=512,
            epochs=100,
            callbacks=[
                EarlyStopping(
                    monitor="val_mean_squared_error", patience=3, verbose=1),
                ModelCheckpoint(
                    model_file_name,
                    save_best_only=True,
                    save_weights_only=True,
                    monitor="val_mean_squared_error",
                    verbose=1),
            ])

        with open(history_file_name, "wb") as history_file:
            pickle.dump(history.history, history_file)

        model.load_weights(model_file_name)
        rmse_valid = np.sqrt(
            model.evaluate(
                np.hsplit(splited_x_valid, x_train.shape[1]), splited_y_valid)[
                    1])
        print("\n\n\033[1m\033[32mRMSE on validation set: {:f}\033[0m\n".
              format(rmse_valid))

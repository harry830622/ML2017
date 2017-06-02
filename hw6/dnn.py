#!/usr/bin/env python3

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate
from keras.layers import Dense, Dropout
from keras.regularizers import l2


def build(num_users=7000,
          num_movies=5000,
          latent_dimension=120,
          is_regularized=True,
          **kwargs):
    lamda = kwargs["lamda"] if "lamda" in kwargs else 1e-5

    user_id_input = Input(shape=(1, ))
    movie_id_input = Input(shape=(1, ))
    # user_feature = Embedding(
    #     users.shape[0],
    #     users.shape[1],
    #     input_length=1,
    #     weights=[users],
    #     trainable=False)(user_id_input)
    # movie_feature = Embedding(
    #     movies.shape[0],
    #     movies.shape[1],
    #     input_length=1,
    #     weights=[movies],
    #     trainable=False)(movie_id_input)
    user_feature = Embedding(
        num_users,
        latent_dimension,
        input_length=1,
        embeddings_initializer="random_uniform",
        embeddings_regularizer=l2(lamda)
        if is_regularized else None)(user_id_input)
    movie_feature = Embedding(
        num_movies,
        latent_dimension,
        input_length=1,
        embeddings_initializer="random_uniform",
        embeddings_regularizer=l2(lamda)
        if is_regularized else None)(movie_id_input)

    user_feature = Flatten()(user_feature)
    movie_feature = Flatten()(movie_feature)
    x = Concatenate()([user_feature, movie_feature])
    x = Dense(256, activation="elu")(x)
    x = Dense(128, activation="elu")(x)
    x = Dense(64, activation="elu")(x)
    output = Dense(1, activation="elu")(x)
    model = Model([user_id_input, movie_id_input], output)

    model.compile(optimizer="adam", loss="mse", metrics=["mse"])

    return model

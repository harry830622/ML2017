#!/usr/bin/env python3

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dot, Add
from keras.regularizers import l2


def build(num_users=7000,
          num_movies=5000,
          latent_dimension=120,
          is_regularized=True,
          is_biased=True,
          **kwargs):
    lamda = kwargs["lamda"] if "lamda" in kwargs else 1e-5
    users = kwargs["users"] if "users" in kwargs else np.empty((0, 0))

    user_id_input = Input(shape=(1, ))
    movie_id_input = Input(shape=(1, ))
    user_latent = Embedding(
        num_users,
        latent_dimension,
        input_length=1,
        embeddings_initializer="random_uniform",
        embeddings_regularizer=l2(lamda)
        if is_regularized else None)(user_id_input)
    movie_latent = Embedding(
        num_movies,
        latent_dimension + users.shape[1],
        input_length=1,
        embeddings_initializer="random_uniform",
        embeddings_regularizer=l2(lamda)
        if is_regularized else None)(movie_id_input)

    if users.shape[0] != 0:
        user_latent = Concatenate()([
            user_latent,
            Embedding(
                num_users,
                users.shape[1],
                input_length=1,
                weights=[users],
                trainable=False)(user_id_input)
        ])

    user_latent = Flatten()(user_latent)
    movie_latent = Flatten()(movie_latent)
    output = Dot(1)([user_latent, movie_latent])
    model = Model([user_id_input, movie_id_input], output)

    if is_biased:
        bias_input = Input(shape=(1, ))
        bias = Embedding(
            1, 1, input_length=1,
            embeddings_initializer="random_uniform")(bias_input)
        user_bias = Embedding(
            num_users,
            1,
            input_length=1,
            embeddings_initializer="random_uniform",
            embeddings_regularizer=l2(lamda)
            if is_regularized else None)(user_id_input)
        movie_bias = Embedding(
            num_movies,
            1,
            input_length=1,
            embeddings_initializer="random_uniform",
            embeddings_regularizer=l2(lamda)
            if is_regularized else None)(movie_id_input)
        bias = Flatten()(bias)
        user_bias = Flatten()(user_bias)
        movie_bias = Flatten()(movie_bias)
        output = Add()([output, bias, user_bias, movie_bias])
        model = Model([user_id_input, movie_id_input, bias_input], output)

    model.compile(optimizer="adam", loss="mse", metrics=["mse"])

    return model

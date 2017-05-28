#!/usr/bin/env python3

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Add
from keras.regularizers import l2


def build(num_users=7000,
          num_movies=5000,
          latent_dimension=120,
          is_regularized=True,
          is_biased=True):
    user_id_input = Input(shape=(1, ))
    movie_id_input = Input(shape=(1, ))
    if is_regularized:
        lamda = 1e-5
        user_latent = Embedding(
            num_users,
            latent_dimension,
            input_length=1,
            embeddings_initializer="random_uniform",
            embeddings_regularizer=l2(lamda))(user_id_input)
        movie_latent = Embedding(
            num_movies,
            latent_dimension,
            input_length=1,
            embeddings_initializer="random_uniform",
            embeddings_regularizer=l2(lamda))(movie_id_input)
    else:
        user_latent = Embedding(
            num_users,
            latent_dimension,
            input_length=1,
            embeddings_initializer="random_uniform")(user_id_input)
        movie_latent = Embedding(
            num_movies,
            latent_dimension,
            input_length=1,
            embeddings_initializer="random_uniform")(movie_id_input)
    user_latent = Flatten()(user_latent)
    movie_latent = Flatten()(movie_latent)
    output = Dot(1)([user_latent, movie_latent])

    if is_biased:
        user_bias = Embedding(
            num_users,
            1,
            input_length=1,
            embeddings_initializer="random_uniform")(user_id_input)
        movie_bias = Embedding(
            num_movies,
            1,
            input_length=1,
            embeddings_initializer="random_uniform")(movie_id_input)
        user_bias = Flatten()(user_bias)
        movie_bias = Flatten()(movie_bias)
        bias = Add()([user_bias, movie_bias])
        output = Add()([output, bias])

    model = Model([user_id_input, movie_id_input], output)

    model.compile(optimizer="adam", loss="mse", metrics=["mse"])

    return model

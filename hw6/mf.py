#!/usr/bin/env python3

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot
from keras.optimizers import Adam


def build(num_users=10000, num_movies=5000, latent_dimension=120):
    user_id_input = Input(shape=(1, ))
    movie_id_input = Input(shape=(1, ))
    user_latent = Embedding(
        num_users, latent_dimension, input_length=1)(user_id_input)
    movie_latent = Embedding(
        num_movies, latent_dimension, input_length=1)(movie_id_input)
    user_latent = Flatten()(user_latent)
    movie_latent = Flatten()(movie_latent)
    output = Dot(1)([user_latent, movie_latent])

    model = Model([user_id_input, movie_id_input], output)

    model.compile(optimizer=Adam(), loss="mse", metrics=["mse"])

    return model

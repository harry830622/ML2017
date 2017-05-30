#!/usr/bin/env python3

import mf

from config import SEED, NUM_MODELS, VALIDATION_RATIO
from config import NUM_USERS, NUM_MOVIES, LATENT_DIMENSION
from config import IS_NORMALIZED, IS_REGULARIZED, LAMBDA, IS_BIASED
from extract import extract_x_test

import numpy as np

import os
import sys
import pickle

pwd = sys.argv[1]
output_file_name = sys.argv[2]
suffix = sys.argv[3]

testing_file_name = os.path.join(pwd, "test.csv")
users_file_name = os.path.join(pwd, "users.csv")
movies_file_name = os.path.join(pwd, "movies.csv")

# users = np.zeros((NUM_USERS, 2 + 7 + 21))

model = mf.build(
    num_users=NUM_USERS,
    num_movies=NUM_MOVIES,
    latent_dimension=LATENT_DIMENSION,
    is_regularized=IS_REGULARIZED,
    lamda=LAMBDA,
    is_biased=IS_BIASED)
model.summary()

x_test = extract_x_test(testing_file_name, is_biased=IS_BIASED)

all_ratings = []
for i in range(NUM_MODELS):
    model_file_name = os.path.join(pwd, "model_{}_{:d}.h5".format(suffix, i))

    model.load_weights(model_file_name)

    all_ratings.append(
        model.predict(
            np.hsplit(x_test, x_test.shape[1]), batch_size=512, verbose=1))
all_ratings = np.array(all_ratings)

if IS_NORMALIZED:
    y_mean_file_name = sys.argv[4]
    with open(y_mean_file_name, "rb") as y_mean_file:
        m = pickle.load(y_mean_file)
        y_mean = m["y_mean"]
        y_mean_index_by_movie_id = m["y_mean_index_by_movie_id"]

    for i in range(all_ratings.shape[0]):
        for j, x in enumerate(x_test):
            movie_id = x[1]
            if movie_id in y_mean_index_by_movie_id:
                all_ratings[i][j] += y_mean[y_mean_index_by_movie_id[movie_id]]
        all_ratings[i][all_ratings[i] < 0] = 0

ratings = np.mean(all_ratings, axis=0)

with open(output_file_name, "w") as output_file:
    output_file.write("TestDataID,Rating\n")
    for i, rating in enumerate(ratings, start=1):
        output_file.write("{:d},{:.3f}\n".format(i, rating[0]))

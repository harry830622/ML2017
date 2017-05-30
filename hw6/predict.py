#!/usr/bin/env python3

import mf

from config import LATENT_DIMENSION
from config import IS_NORMALIZED, IS_REGULARIZED, IS_BIASED
from extract import extract_x_test

import numpy as np

import sys
import pickle

testing_file_name = sys.argv[1]
output_file_name = sys.argv[2]
model_file_name = sys.argv[3]

model = mf.build(
    latent_dimension=LATENT_DIMENSION,
    is_regularized=IS_REGULARIZED,
    is_biased=IS_BIASED)
model.summary()

model.load_weights(model_file_name)

x_test = extract_x_test(testing_file_name, is_biased=IS_BIASED)

ratings = model.predict(
    np.hsplit(x_test, x_test.shape[1]), batch_size=512, verbose=1)

if IS_NORMALIZED:
    y_mean_file_name = sys.argv[4]
    with open(y_mean_file_name, "rb") as y_mean_file:
        m = pickle.load(y_mean_file)
        y_mean = m["y_mean"]
        y_mean_index_by_movie_id = m["y_mean_index_by_movie_id"]
    for i, x in enumerate(x_test):
        movie_id = x[1]
        if movie_id in y_mean_index_by_movie_id:
            ratings[i] += y_mean[y_mean_index_by_movie_id[movie_id]]
    ratings[ratings < 0] = 0

with open(output_file_name, "w") as output_file:
    output_file.write("TestDataID,Rating\n")
    for i, rating in enumerate(ratings, start=1):
        output_file.write("{:d},{:.3f}\n".format(i, rating[0]))

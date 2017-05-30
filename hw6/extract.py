#!/usr/bin/env python3

import numpy as np

import pickle


def extract_xy_train(training_file_name, is_normalized=True, is_biased=True):
    x_train = []
    y_train = []
    y_mean = []
    y_mean_indices = []
    y_mean_index_by_movie_id = {}
    with open(training_file_name, "r") as training_file:
        nth_line = 0
        for line in training_file:
            nth_line += 1
            if nth_line != 1:
                _, user_id, movie_id, rating = [
                    int(s) for s in line.strip("\n").split(",")
                ]
                x_train.append([user_id, movie_id])
                y_train.append(rating)
                current_y_mean_index = -1
                if movie_id in y_mean_index_by_movie_id:
                    current_y_mean_index = y_mean_index_by_movie_id[movie_id]
                    y_mean[current_y_mean_index].append(rating)
                else:
                    current_y_mean_index = len(y_mean)
                    y_mean.append([rating])
                    y_mean_index_by_movie_id[movie_id] = current_y_mean_index
                y_mean_indices.append(current_y_mean_index)

    if is_normalized:
        for i, m in enumerate(y_mean):
            average_rating = sum(m) / len(m)
            y_mean[i] = average_rating
        for i, y_mean_index in enumerate(y_mean_indices):
            y_train[i] -= y_mean[y_mean_index]

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    if is_biased:
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))

    y_mean_file_name = "y_mean.p"
    with open(y_mean_file_name, "wb") as y_mean_file:
        pickle.dump({
            "y_mean": y_mean,
            "y_mean_index_by_movie_id": y_mean_index_by_movie_id,
        }, y_mean_file)

    return x_train, y_train


def extract_x_test(testing_file_name, is_biased=True):
    x_test = []
    with open(testing_file_name, "r") as testing_file:
        nth_line = 0
        for line in testing_file:
            nth_line += 1
            if nth_line != 1:
                _, user_id, movie_id = [
                    int(s) for s in line.strip("\n").split(",")
                ]
                x_test.append([user_id, movie_id])
    x_test = np.array(x_test)

    if is_biased:
        x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

    return x_test

#!/usr/bin/env python3

from config import NUM_MOVIES

import numpy as np

import pickle


def extract_xy_train(training_file_name, is_normalized=True, is_biased=True):
    x_train = []
    y_train = []
    y_mean = [[] for i in range(NUM_MOVIES)]
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
                y_mean[movie_id].append(rating)

    for i, ratings in enumerate(y_mean):
        y_mean[i] = sum(ratings) / len(ratings) if len(ratings) != 0 else 0

    if is_normalized:
        for i, y in enumerate(y_train):
            movie_id = x_train[i][1]
            y_train[i] -= y_mean[movie_id]

    x_train = np.array(x_train, dtype=np.int64)
    y_train = np.array(y_train, dtype=np.float64)

    if is_biased:
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))

    y_mean_file_name = "y_mean.p"
    with open(y_mean_file_name, "wb") as y_mean_file:
        pickle.dump(y_mean, y_mean_file)

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
    x_test = np.array(x_test, dtype=np.int64)

    if is_biased:
        x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

    return x_test


def extract_users(users_file_name, num_users=7000):
    users = [[0 for j in range(2 + 7 + 21)] for i in range(num_users)]
    with open(users_file_name, "r") as users_file:
        nth_line = 0
        for line in users_file:
            nth_line += 1
            if nth_line != 1:
                user_id, gender, age, occupation, _ = line.strip("\n").split(
                    "::")
                gender = [1, 0] if gender == "M" else [0, 1]
                age = [
                    1 if age == s else 0
                    for s in ["1", "18", "25", "35", "45", "50", "56"]
                ]
                occupation = [
                    1 if occupation == str(i) else 0 for i in range(21)
                ]
                users[int(user_id)] = gender + age + occupation
    users = np.array(users, dtype=np.int64)

    return users


def extract_movies(movies_file_name, num_movies=10000):
    movies = [[0 for j in range(18)] for i in range(num_movies)]
    with open(movies_file_name, "r", encoding="latin_1") as movies_file:
        nth_line = 0
        for line in movies_file:
            nth_line += 1
            if nth_line != 1:
                movie_id, _, genres = line.strip("\n").split("::")
                genres = [
                    1 if s in genres.split("|") else 0
                    for s in [
                        "Action", "Adventure", "Animation", "Children's",
                        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                        "Sci-Fi", "Thriller", "War", "Western"
                    ]
                ]
                movies[int(movie_id)] = genres
    movies = np.array(movies, dtype=np.int64)

    return movies

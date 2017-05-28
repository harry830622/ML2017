#!/usr/bin/env python3


def extract_training_xy(training_file_name):
    training_x = []
    training_y = []
    with open(training_file_name, "r") as training_file:
        nth_line = 0
        for line in training_file:
            nth_line += 1
            if nth_line != 1:
                _, user_id, movie_id, rating = [
                    int(s) for s in line.strip("\n").split(",")
                ]
                training_x.append([user_id, movie_id])
                training_y.append(rating)
    return training_x, training_y


def extract_testing_x(testing_file_name):
    testing_x = []
    with open(testing_file_name, "r") as testing_file:
        nth_line = 0
        for line in testing_file:
            nth_line += 1
            if nth_line != 1:
                _, user_id, movie_id = [
                    int(s) for s in line.strip("\n").split(",")
                ]
                testing_x.append([user_id, movie_id])
    return testing_x

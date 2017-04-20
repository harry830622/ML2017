#!/usr/bin/env python

import csv
import sys
import pickle

training_file_name = sys.argv[1]
testing_file_name = sys.argv[2]
training_x_dump_name = sys.argv[3]
training_y_dump_name = sys.argv[4]
testing_x_dump_name = sys.argv[5]

training_x = []
training_y = []
with open(training_file_name, "r") as training_file:
    training_csv = csv.reader(training_file)
    nth_row = 0
    for row in training_csv:
        nth_row += 1
        if nth_row != 1:
            label, pixels = row
            training_x.append([int(pixel) for pixel in pixels.split(" ")])
            training_y.append(int(label))

testing_x = []
with open(testing_file_name, "r") as testing_file:
    testing_csv = csv.reader(testing_file)
    nth_row = 0
    for row in testing_csv:
        nth_row += 1
        if nth_row != 1:
            _, pixels = row
            testing_x.append([int(pixel) for pixel in pixels.split(" ")])

with open(training_x_dump_name, "wb") as training_x_dump:
    pickle.dump(training_x, training_x_dump)

with open(training_y_dump_name, "wb") as training_y_dump:
    pickle.dump(training_y, training_y_dump)

with open(testing_x_dump_name, "wb") as testing_x_dump:
    pickle.dump(testing_x, testing_x_dump)

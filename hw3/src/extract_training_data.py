#!/usr/bin/env python

import csv
import sys
import pickle

training_file_name = sys.argv[1]
training_x_dump_name = sys.argv[2]
training_y_dump_name = sys.argv[3]

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

with open(training_x_dump_name, "wb") as training_x_dump:
    pickle.dump(training_x, training_x_dump)

with open(training_y_dump_name, "wb") as training_y_dump:
    pickle.dump(training_y, training_y_dump)

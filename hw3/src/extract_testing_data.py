#!/usr/bin/env python

import csv
import sys
import pickle

testing_file_name = sys.argv[1]
testing_x_dump_name = sys.argv[2]

testing_x = []
with open(testing_file_name, "r") as testing_file:
    testing_csv = csv.reader(testing_file)
    nth_row = 0
    for row in testing_csv:
        nth_row += 1
        if nth_row != 1:
            _, pixels = row
            testing_x.append([int(pixel) for pixel in pixels.split(" ")])

with open(testing_x_dump_name, "wb") as testing_x_dump:
    pickle.dump(testing_x, testing_x_dump)

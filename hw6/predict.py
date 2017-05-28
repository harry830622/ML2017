#!/usr/bin/env python3

import mf

from extract import extract_testing_x

import numpy as np

import sys
import pickle

testing_file_name = sys.argv[1]
output_file_name = sys.argv[2]
model_file_name = sys.argv[3]

model = mf.build()
model.summary()

model.load_weights(model_file_name)

testing_x = extract_testing_x(testing_file_name)
testing_x = np.array(testing_x)

ratings = model.predict(np.hsplit(testing_x, 2))

with open(output_file_name, "w") as output_file:
    output_file.write("TestDataID,Rating\n")
    for i, rating in enumerate(ratings, start=1):
        output_file.write("{:d},{:.3f}\n".format(i, rating[0]))

#!/usr/bin/env python

import feature_extractor

import numpy as np
import sys

def sigmoid(z):
    sig = 1 / (1.0 + np.exp(-z))
    return np.clip(sig, 0.00000000000001, 0.99999999999999)

training_x, training_y, testing_x = feature_extractor.extract_features(sys.argv[1], sys.argv[2])


output_file_name = sys.argv[6]
with open(output_file_name, "w") as output_file:
    output_file.write("id,label\n")
    i = 0
    for row in testing_x:
        i += 1
        y = sigmoid()
        output_file.write(str(i) + ",")
        output_file.write("1\n" if y >= 0.5 else "0\n")

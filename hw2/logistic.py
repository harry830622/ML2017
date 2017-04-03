#!/usr/bin/env python

import feature_extractor

import numpy as np
import sys

def sigmoid(z):
    sig = 1 / (1.0 + np.exp(-z))
    return np.clip(sig, 0.00000000000001, 0.99999999999999)

training_x, training_y, testing_x = feature_extractor.extract_features(sys.argv[1], sys.argv[2])

num_features = training_x.shape[1]
weights = [ 0.0 for _ in range(num_features) ]

num_iterations = 1e6
learning_rate = 1

t = 0
previous_gradient = [ 0.0 for _ in range(num_features) ]
weights = np.matrix([weights], dtype = np.float64).transpose()
previous_gradient = np.matrix([previous_gradient], dtype = np.float64).transpose()
while t < num_iterations:
    t += 1
    y = sigmoid(training_x.dot(weights))
    loss = y - training_y
    gradient = training_x.transpose().dot(loss)
    previous_gradient += np.square(gradient)
    weights -= learning_rate * gradient / np.sqrt(previous_gradient)
    if t % 100 == 0:
        print("ERR:", np.sum(np.abs(loss)))

output_file_name = sys.argv[6]
with open(output_file_name, "w") as output_file:
    output_file.write("id,label\n")
    i = 0
    for row in testing_x:
        i += 1
        y = sigmoid(row.dot(weights))
        output_file.write(str(i) + ",")
        output_file.write("1\n" if y >= 0.5 else "0\n")

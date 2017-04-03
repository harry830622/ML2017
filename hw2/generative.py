#!/usr/bin/env python

import feature_extractor

import numpy as np
import sys

feature_config = {
    "age": True,
    "workclass": True,
    "fnlwgt": False,
    "education": True,
    "education-num": True,
    "marital-status": True,
    "occupation": True,
    "relationship": True,
    "race": True,
    "sex": True,
    "capital-gain": True,
    "capital-loss": True,
    "hours-per-week": True,
    "native-country": True,
}

def sigmoid(z):
    sig = 1 / (1.0 + np.exp(-z))
    return np.clip(sig, 0.00000000000001, 0.99999999999999)

training_x, training_y, testing_x = feature_extractor.extract_features(sys.argv[1], sys.argv[2], feature_config, True)

num_training_data = training_x.shape[0]
num_features = training_x.shape[1]

true_training_x = []
false_training_x = []
for i, row in enumerate(training_x.tolist()):
    if training_y[i, 0] == 1.0:
        true_training_x.append(row)
    else:
        false_training_x.append(row)

true_training_x = np.matrix(true_training_x, dtype = np.float64)
false_training_x = np.matrix(false_training_x, dtype = np.float64)

u1 = true_training_x.mean(axis = 0)
u2 = false_training_x.mean(axis = 0)

sigma1 = np.matrix(np.zeros((num_features, num_features)), dtype = np.float64)
sigma2 = np.matrix(np.zeros((num_features, num_features)), dtype = np.float64)
for row in true_training_x:
    sigma1 += (row - u1).transpose().dot(row - u1)
for row in false_training_x:
    sigma2 += (row - u2).transpose().dot(row - u2)
num_true = true_training_x.shape[0]
num_false = false_training_x.shape[0]
sigma1 /= num_true
sigma2 /= num_false
shared_sigma = num_true / float(num_training_data) * sigma1 + num_false / float(num_training_data) * sigma2
shared_sigma_inverse = shared_sigma.getI()
weights = (u1 - u2).dot(shared_sigma_inverse).transpose()
bias = (-0.5) * u1.dot(shared_sigma_inverse).dot(u1.transpose()) + 0.5 * u2.dot(shared_sigma_inverse).dot(u2.transpose()) + np.log(num_true / float(num_false))

output_file_name = sys.argv[6]
with open(output_file_name, "w") as output_file:
    output_file.write("id,label\n")
    i = 0
    for row in testing_x:
        i += 1
        y = sigmoid(row.dot(weights) + bias[0, 0])
        output_file.write(str(i) + ",")
        output_file.write("1\n" if y >= 0.5 else "0\n")

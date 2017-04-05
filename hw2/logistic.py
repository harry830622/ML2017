#!/usr/bin/env python

import feature_extractor

import numpy as np
import sys

feature_config = {
    "bias": [1],
    "age": [1, 2],
    "fnlwgt": [1, 2],
    "education-num": [1, 2],
    "capital-gain": [1, 2],
    "capital-loss": [1, 2],
    "hours-per-week": [1, 2],
    "workclass": [0],
    "education": [0],
    "marital-status": [0],
    "occupation": [0],
    "relationship": [0],
    "race": [0],
    "sex": [0],
    "native-country": [0],
}


def sigmoid(z):
    sig = 1 / (1.0 + np.exp(-z))
    return np.clip(sig, 0.00000000000001, 0.99999999999999)


def err(loss):
    return np.sum(np.abs(loss))


def cross_entropy(y_hat, y):
    return np.sum(-1 * y_hat.transpose().dot(np.log(sigmoid(y))) - (
        1 - y_hat).transpose().dot(1 - np.log(sigmoid(y))))


training_x, training_y, testing_x = feature_extractor.extract_features(
    sys.argv[1], sys.argv[2], feature_config, True)

num_validating_data = training_x.shape[0] // 10
validating_x = training_x[:num_validating_data]
validating_y = training_y[:num_validating_data]
training_x = training_x[num_validating_data:]
training_y = training_y[num_validating_data:]
num_training_data = training_x.shape[0]

num_features = training_x.shape[1]
weights = [0.0 for _ in range(num_features)]

# Training parameters
num_iterations = 1e4
learning_rate = 1e1
is_regularized = True
lamda = 1e1

print("========== Training ==========")
t = 0
previous_gradient = [0.0 for _ in range(num_features)]
weights = np.matrix([weights], dtype=np.float64).transpose()
previous_gradient = np.matrix(
    [previous_gradient], dtype=np.float64).transpose()
while t < num_iterations:
    t += 1
    y = sigmoid(training_x.dot(weights))
    loss = y - training_y
    gradient = training_x.transpose().dot(loss)
    if is_regularized:
        regularizer = lamda * np.sum(weights)
        gradient += regularizer
        loss += regularizer
    previous_gradient += np.square(gradient)
    weights -= learning_rate * gradient / np.sqrt(previous_gradient)
    if t % 100 == 0:
        print("ERR: %f \tL: %f" % (err(loss) / float(num_training_data),
                                   cross_entropy(training_y, y)))

print("========== Validating ==========")
y = sigmoid(validating_x.dot(weights))
loss = y - validating_y
accuracy = 1.0 - np.sum(np.abs(
    (y + 0.5) // 1 - validating_y)) / num_validating_data
print("ERR: %f \tL: %f" % (err(loss) / float(num_validating_data),
                           cross_entropy(validating_y, y)))
print("ACCURACY: %f%%" % (accuracy * 100))

output_file_name = sys.argv[6]
with open(output_file_name, "w") as output_file:
    output_file.write("id,label\n")
    i = 0
    for row in testing_x:
        i += 1
        y = sigmoid(row.dot(weights))
        output_file.write(str(i) + ",")
        output_file.write("1\n" if y >= 0.5 else "0\n")

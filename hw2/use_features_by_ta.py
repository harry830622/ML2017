import numpy as np
import csv
import sys

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

raw_training_file_name = sys.argv[1]
raw_testing_file_name = sys.argv[2]
training_x_file_name = sys.argv[3]
training_y_file_name = sys.argv[4]
testing_x_file_name = sys.argv[5]
output_file_name = sys.argv[6]

feature_names = []
training_x = []
with open(training_x_file_name, "r") as training_x_file:
    training_x_csv = csv.reader(training_x_file)
    nth_row = 0
    for row in training_x_csv:
        nth_row += 1
        if nth_row == 1:
            feature_names = row
        else:
            training_x.append(
                    [1] + [ float(s) for i, s in enumerate(row) if i != 1])

training_y = []
with open(training_y_file_name, "r") as training_y_file:
    for line in training_y_file:
        training_y.append(float(line))

num_features = len(training_x[0])
weights = [ 0.0 for _ in range(num_features) ]

num_iterations = 1e6
learning_rate = 1e-2

t = 0
previous_gradient = [ 0.0 for _ in range(num_features) ]
training_x = np.matrix(training_x, dtype = np.float64)
training_y = np.matrix([training_y], dtype = np.float64).transpose()
weights = np.matrix([weights], dtype = np.float64).transpose()
previous_gradient = np.matrix([previous_gradient], dtype = np.float64).transpose()
while t < num_iterations:
    t += 1
    y = sigmoid(np.dot(training_x, weights))
    loss = y - training_y
    gradient = np.dot(np.transpose(training_x), loss)
    previous_gradient += np.square(gradient)
    weights -= learning_rate * gradient / np.sqrt(previous_gradient)
    if t % 100 == 0:
        print("ERR:", np.sum(np.abs(loss)))

testing_x = []
with open(testing_x_file_name, "r") as testing_x_file:
    testing_x_csv = csv.reader(testing_x_file)
    nth_row = 0
    for row in testing_x_csv:
        nth_row += 1
        if nth_row != 1:
            testing_x.append(
                    [1] + [ float(s) for i, s in enumerate(row) if i != 1])

with open(output_file_name, "w") as output_file:
    output_file.write("id,label\n")
    i = 0
    for row in testing_x:
        i += 1
        y = sigmoid(np.dot(row, weights))
        output_file.write(str(i) + ",")
        output_file.write("1\n" if y >= 0.5 else "0\n")

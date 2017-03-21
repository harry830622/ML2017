import numpy as np
import csv
import sys

index_names = [
        "AMB_TEMP",
        "CH4",
        "CO",
        "NMHC",
        "NO",
        "NO2",
        "NOx",
        "O3",
        "PM10",
        "PM2.5",
        "RAINFALL",
        "RH",
        "SO2",
        "THC",
        "WD_HR",
        "WIND_DIREC",
        "WIND_SPEED",
        "WS_HR",
        ]

training_file_name = sys.argv[1]
testing_file_name = sys.argv[2]
output_file_name = sys.argv[3]

training_data = [ [] for i in range(18) ]
with open(training_file_name, "r", encoding = "big5") as training_file:
    training_csv = csv.reader(training_file)
    nth_row = 0
    for row in training_csv:
        if nth_row != 0:
            index_values = [ float(s) if s != "NR" else 0.0 for s in row[3:] ]
            training_data[(nth_row - 1) % 18] += index_values
        nth_row += 1

training_x = []
training_y = []
num_month_data = 24 * 20
for i in range(12):
    for j in range(num_month_data - 9):
        training_x.append([1])
        for k in range(18):
            for h in range(9):
                training_x[(num_month_data - 9) * i + j].append(
                        training_data[k][num_month_data * i + j + h])
        training_y.append(training_data[9][num_month_data * i + j + 9])

feature_table = np.matrix([[
    1, # bias
    0, 0, 0, 0, 0, 0, 0, 0, 0, # AMB_TEMP
    0, 0, 0, 0, 0, 0, 0, 0, 0, # CH4
    0, 0, 0, 0, 0, 0, 0, 0, 0, # CO
    0, 0, 0, 0, 0, 0, 0, 0, 0, # NMHC
    0, 0, 0, 0, 0, 0, 0, 0, 0, # NO
    0, 0, 0, 0, 0, 0, 0, 0, 0, # NO2
    0, 0, 0, 0, 0, 0, 0, 0, 0, # NOx
    1, 1, 1, 1, 1, 1, 1, 1, 1, # O3
    1, 1, 1, 1, 1, 1, 1, 1, 1, # PM10
    1, 1, 1, 1, 1, 1, 1, 1, 1, # PM2.5
    1, 1, 1, 1, 1, 1, 1, 1, 1, # RAINFALL
    0, 0, 0, 0, 0, 0, 0, 0, 0, # RH
    0, 0, 0, 0, 0, 0, 0, 0, 0, # SO2
    0, 0, 0, 0, 0, 0, 0, 0, 0, # THC
    1, 1, 1, 1, 1, 1, 1, 1, 1, # WD_HR
    0, 0, 0, 0, 0, 0, 0, 0, 0, # WIND_DIREC
    1, 1, 1, 1, 1, 1, 1, 1, 1, # WIND_SPEED
    1, 1, 1, 1, 1, 1, 1, 1, 1, # WS_HR
    ]]).transpose()

num_features = 1 + 18 * 9 # Includes bias.
weights = np.matrix([[ 0.0 for i in range(num_features) ]]).transpose()
# print(weights.shape)
num_iterations = 1000
learning_rate = 1000
previous_gradient = np.matrix(
        [[ 0.0 for i in range(num_features)]]).transpose()
previous_RMSE = 0.0
for t in range(num_iterations):
    y = np.dot(np.matrix(training_x), weights)
    loss_root = y - np.matrix(training_y).transpose()
    gradient = 2 * np.dot(np.transpose(np.matrix(training_x)), loss_root)
    # print(gradient.shape)
    previous_gradient += np.square(gradient)
    weights -= learning_rate * (gradient / np.sqrt(previous_gradient))
    weights = np.multiply(weights, feature_table)
    # learning_rate = learning_rate / (t + 1) ** 0.5
    # print(weights)
    # sys.stdin.read(1)
    RMSE = (np.sum(np.square(loss_root)) / (num_month_data - 9) / 12) ** 0.5
    if t % 100 == 0:
        print(RMSE)
    # if abs(previous_RMSE - RMSE) < 1e-7:
    #     break
    previous_RMSE = RMSE

test_x = {}
test_y = {}
with open(testing_file_name, "r", encoding = "big5") as testing_file:
    testing_csv = csv.reader(testing_file)
    i = 1
    for row in testing_csv:
        d = row[0]
        if not d in test_x:
            test_x[d] = [1]
        test_x[d] += [ float(s) if s != "NR" else 0.0 for s in row[2:] ]
        if i % 18 == 0:
            test_y[d] = np.dot(np.matrix([test_x[d]]), weights)
        i += 1

output_string = "id,value\n"
for k, y in test_y.items():
    output_string += (k + "," + str(y.item(0)) + "\n")

with open(output_file_name, "w") as f:
    f.write(output_string)

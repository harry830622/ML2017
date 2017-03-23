import numpy as np
import csv
import copy
import sys
import os.path
import pickle

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

feature_table = [
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
    ]

training_file_name = sys.argv[1]
testing_file_name = sys.argv[2]
output_file_name = sys.argv[3]

model_file_name = "./model"
is_model_existed = os.path.isfile(model_file_name)

mean = []
max_min = []
log = {
        "feature_table": feature_table,
        "mean": mean,
        "max_min": max_min,
        "weights": [],
        "RMSEs": [],
        }

# is_model_existed = False
if not is_model_existed:
    print("Model doesn't exist. Start training...")

    training_data = [ [] for i in range(18) ]
    with open(training_file_name, "r", encoding = "big5") as training_file:
        training_csv = csv.reader(training_file)
        nth_row = 0
        for row in training_csv:
            if nth_row != 0:
                index_values = [ float(
                    s) if s != "NR" else 0.0 for s in row[3:] ]
                training_data[(nth_row - 1) % 18] += index_values
            nth_row += 1

    raw_data = copy.deepcopy(training_data)

    training_data = np.matrix(training_data)
    mean = training_data.mean(1)
    max_min = training_data.max(1) - training_data.min(1)
    training_data = (training_data - mean) / max_min

    log["mean"] = mean.flatten().tolist()[0]
    log["max_min"] = max_min.flatten().tolist()[0]

    training_data = training_data.tolist()

    training_x = []
    training_y = []
    num_months = 12
    num_month_data = 20 * 24
    for i in range(num_months):
        for j in range(num_month_data - 9):
            training_x.append([1])
            for k in range(18):
                for h in range(9):
                    training_x[(num_month_data - 9) * i + j].append(
                            training_data[k][num_month_data * i + j + h])
            training_y.append(raw_data[9][num_month_data * i + j + 9])

    # validating_x = []
    # validating_y = []
    # for i in range(2):
    #     for j in range(num_month_data - 9):
    #         validating_x.append([1])
    #         for k in range(18):
    #             for h in range(9):
    #                 validating_x[(num_month_data - 9) * i + j].append(
    #                         training_data[k][num_month_data * (i + 10) + j + h])
    #         validating_y.append(raw_data[9][num_month_data * (i + 10) + j + 9])

    for i, features in enumerate(training_x):
        training_x[i] = [ n for j, n in enumerate(
            features) if feature_table[j] == 1 ]
    # for i, features in enumerate(validating_x):
    #     validating_x[i] = [ n for j, n in enumerate(
    #         features) if feature_table[j] == 1 ]

    # for features in training_x:
    #     features += [ n ** 2 for n in features ]
    # for features in validating_x:
    #     features += [ n ** 2 for n in features ]
    # for features in training_x:
    #     features += [ n ** 3 for n in features ]
    # for features in validating_x:
    #     features += [ n ** 3 for n in features ]
    # for features in training_x:
    #     features += [ n ** 4 for n in features ]
    # for features in validating_x:
    #     features += [ n ** 4 for n in features ]

    num_features = len(training_x[0])
    weights = np.matrix([[ 0.0 for _ in range(num_features) ]]).transpose()
    num_iterations = 1000000
    learning_rate = 1000
    # lamda = 100000
    previous_gradient = np.matrix(
            [[ 0.0 for _ in range(num_features)]]).transpose()
    previous_RMSE = 0.0
    for t in range(num_iterations):
        y = np.dot(np.matrix(training_x), weights)
        loss_root = y - np.matrix(training_y).transpose()
        # regularizer = np.copy(weights)
        # regularizer[0, 0] = 0
        # regularizer = np.sum(regularizer)
        # gradient = 2 * np.dot(np.matrix(training_x).transpose(), loss_root) + (
        #         2 * lamda * regularizer)
        gradient = 2 * np.dot(np.matrix(training_x).transpose(), loss_root)
        previous_gradient += np.square(gradient)
        weights -= learning_rate * (gradient / np.sqrt(previous_gradient))
        RMSE = (np.sum(np.square(loss_root)) / (
            num_month_data - 9) / num_months) ** 0.5
        if t % 100 == 0:
            log["RMSEs"].append(RMSE)
            print("RMSE:", RMSE)
        if abs(RMSE - previous_RMSE) < 1e-11:
            print("RMSE:", RMSE)
            log["RMSEs"].append(RMSE)
            break
        previous_RMSE = RMSE
    print("Final RMSE:", previous_RMSE)
    log["RMSEs"].append(previous_RMSE)
    log["weights"] = weights.flatten().tolist()[0]

    # v_y = np.dot(np.matrix(validating_x), weights)
    # v_loss_root = v_y - np.matrix(validating_y).transpose()
    # v_RMSE = (np.sum(np.square(v_loss_root)) / (num_month_data - 9) / 2) ** 0.5
    # print("Validating RMSE:", v_RMSE)

    with open("model", "wb") as model:
        pickle.dump(log, model)

else:
    print("Model exists. Use trained weights...")
    with open(model_file_name, "rb") as model:
        log = pickle.load(model)
        feature_table = log["feature_table"]
        weights = log["weights"]
        print("Features:", feature_table)
        print("Trained weights:", weights)
        mean = np.matrix(log["mean"]).transpose()
        max_min = np.matrix(log["max_min"]).transpose()
        weights = np.matrix(weights).transpose()

test_x = {}
test_y = {}
with open(testing_file_name, "r", encoding = "big5") as testing_file:
    testing_csv = csv.reader(testing_file)
    i = 0
    for row in testing_csv:
        d = row[0]
        if not d in test_x:
            test_x[d] = [1]
        nine_hours = (np.matrix(
                [[ float(s) if s != "NR" else 0.0 for s in row[2:] ]]) - (
                        mean.item(i % 18))) / max_min.item(i % 18)
        test_x[d] += nine_hours.tolist()[0]
        i += 1

for k, xs in test_x.items():
    test_x[k] = [ n for j, n in enumerate(xs) if feature_table[j] == 1 ]
    test_y[k] = np.dot(np.matrix([test_x[k]]), weights)

output_string = "id,value\n"
for k, y in test_y.items():
    output_string += (k + "," + str(y.item(0)) + "\n")

with open(output_file_name, "w") as f:
    f.write(output_string)

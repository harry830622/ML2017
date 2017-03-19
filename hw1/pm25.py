from model import Model

import sys
import csv
import math

training_file_name = sys.argv[1]
testing_file_name = sys.argv[2]
output_file_name = sys.argv[3]

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

training_data = { k: [ [] for i in range(12) ] for k in index_names }
nth_index = 0
nth_day = 0
nth_month = 0
with open(training_file_name, "r", encoding = "big5") as training_file:
    training_csv = csv.reader(training_file)
    for row in training_csv:
        index_name = row[2]
        if index_name in index_names:
            index_values = [ float(s) if s != "NR" else 0.0 for s in row[3:] ]
            training_data[index_name][nth_month] += index_values
            nth_index += 1
            if nth_index == len(index_names):
                nth_index = 0
                nth_day += 1
                if nth_day == 20:
                    nth_day = 0
                    nth_month += 1

# print(training_data)

freeze_gradient = 0.0001
num_iterations = 1000000
num_examples = 1
learning_rate = 0.001
bias = 0.0
feature_config = {
        "PM2.5-1^1": 0.0,
        "PM2.5-2^1": 0.0,
        "PM2.5-3^1": 0.0,
        "PM2.5-4^1": 0.0,
        "PM2.5-5^1": 0.0,
        "PM2.5-6^1": 0.0,
        "PM2.5-7^1": 0.0,
        "PM2.5-8^1": 0.0,
        "PM2.5-9^1": 0.0,
        # "AMB_TEMP-1^1": 0.0,
        # "AMB_TEMP-2^1": 0.0,
        # "AMB_TEMP-3^1": 0.0,
        # "AMB_TEMP-4^1": 0.0,
        # "AMB_TEMP-5^1": 0.0,
        # "AMB_TEMP-6^1": 0.0,
        # "AMB_TEMP-7^1": 0.0,
        # "AMB_TEMP-8^1": 0.0,
        # "AMB_TEMP-9^1": 0.0,
        # "CH4-1^1": 0.0,
        # "CH4-2^1": 0.0,
        # "CH4-3^1": 0.0,
        # "CH4-4^1": 0.0,
        # "CH4-5^1": 0.0,
        # "CH4-6^1": 0.0,
        # "CH4-7^1": 0.0,
        # "CH4-8^1": 0.0,
        # "CH4-9^1": 0.0,
        # "CO-1^1": 0.0,
        # "CO-2^1": 0.0,
        # "CO-3^1": 0.0,
        # "CO-4^1": 0.0,
        # "CO-5^1": 0.0,
        # "CO-6^1": 0.0,
        # "CO-7^1": 0.0,
        # "CO-8^1": 0.0,
        # "CO-9^1": 0.0,
        # "NMHC-1^1": 0.0,
        # "NMHC-2^1": 0.0,
        # "NMHC-3^1": 0.0,
        # "NMHC-4^1": 0.0,
        # "NMHC-5^1": 0.0,
        # "NMHC-6^1": 0.0,
        # "NMHC-7^1": 0.0,
        # "NMHC-8^1": 0.0,
        # "NMHC-9^1": 0.0,
        # "NO-1^1": 0.0,
        # "NO-2^1": 0.0,
        # "NO-3^1": 0.0,
        # "NO-4^1": 0.0,
        # "NO-5^1": 0.0,
        # "NO-6^1": 0.0,
        # "NO-7^1": 0.0,
        # "NO-8^1": 0.0,
        # "NO-9^1": 0.0,
        # "NO2-1^1": 0.0,
        # "NO2-2^1": 0.0,
        # "NO2-3^1": 0.0,
        # "NO2-4^1": 0.0,
        # "NO2-5^1": 0.0,
        # "NO2-6^1": 0.0,
        # "NO2-7^1": 0.0,
        # "NO2-8^1": 0.0,
        # "NO2-9^1": 0.0,
        # "NOx-1^1": 0.0,
        # "NOx-2^1": 0.0,
        # "NOx-3^1": 0.0,
        # "NOx-4^1": 0.0,
        # "NOx-5^1": 0.0,
        # "NOx-6^1": 0.0,
        # "NOx-7^1": 0.0,
        # "NOx-8^1": 0.0,
        # "NOx-9^1": 0.0,
        # "O3-1^1": 0.0,
        # "O3-2^1": 0.0,
        # "O3-3^1": 0.0,
        # "O3-4^1": 0.0,
        # "O3-5^1": 0.0,
        # "O3-6^1": 0.0,
        # "O3-7^1": 0.0,
        # "O3-8^1": 0.0,
        # "O3-9^1": 0.0,
        # "PM10-1^1": 0.0,
        # "PM10-2^1": 0.0,
        # "PM10-3^1": 0.0,
        # "PM10-4^1": 0.0,
        # "PM10-5^1": 0.0,
        # "PM10-6^1": 0.0,
        # "PM10-7^1": 0.0,
        # "PM10-8^1": 0.0,
        # "PM10-9^1": 0.0,
        # "RAINFALL-1^1": 0.0,
        # "RAINFALL-2^1": 0.0,
        # "RAINFALL-3^1": 0.0,
        # "RAINFALL-4^1": 0.0,
        # "RAINFALL-5^1": 0.0,
        # "RAINFALL-6^1": 0.0,
        # "RAINFALL-7^1": 0.0,
        # "RAINFALL-8^1": 0.0,
        # "RAINFALL-9^1": 0.0,
        # "RH-1^1": 0.0,
        # "RH-2^1": 0.0,
        # "RH-3^1": 0.0,
        # "RH-4^1": 0.0,
        # "RH-5^1": 0.0,
        # "RH-6^1": 0.0,
        # "RH-7^1": 0.0,
        # "RH-8^1": 0.0,
        # "RH-9^1": 0.0,
        # "SO2-1^1": 0.0,
        # "SO2-2^1": 0.0,
        # "SO2-3^1": 0.0,
        # "SO2-4^1": 0.0,
        # "SO2-5^1": 0.0,
        # "SO2-6^1": 0.0,
        # "SO2-7^1": 0.0,
        # "SO2-8^1": 0.0,
        # "SO2-9^1": 0.0,
        # "THC-1^1": 0.0,
        # "THC-2^1": 0.0,
        # "THC-3^1": 0.0,
        # "THC-4^1": 0.0,
        # "THC-5^1": 0.0,
        # "THC-6^1": 0.0,
        # "THC-7^1": 0.0,
        # "THC-8^1": 0.0,
        # "THC-9^1": 0.0,
        # "WD_HR-1^1": 0.0,
        # "WD_HR-2^1": 0.0,
        # "WD_HR-3^1": 0.0,
        # "WD_HR-4^1": 0.0,
        # "WD_HR-5^1": 0.0,
        # "WD_HR-6^1": 0.0,
        # "WD_HR-7^1": 0.0,
        # "WD_HR-8^1": 0.0,
        # "WD_HR-9^1": 0.0,
        # "WIND_DIREC-1^1": 0.0,
        # "WIND_DIREC-2^1": 0.0,
        # "WIND_DIREC-3^1": 0.0,
        # "WIND_DIREC-4^1": 0.0,
        # "WIND_DIREC-5^1": 0.0,
        # "WIND_DIREC-6^1": 0.0,
        # "WIND_DIREC-7^1": 0.0,
        # "WIND_DIREC-8^1": 0.0,
        # "WIND_DIREC-9^1": 0.0,
        # "WIND_SPEED-1^1": 0.0,
        # "WIND_SPEED-2^1": 0.0,
        # "WIND_SPEED-3^1": 0.0,
        # "WIND_SPEED-4^1": 0.0,
        # "WIND_SPEED-5^1": 0.0,
        # "WIND_SPEED-6^1": 0.0,
        # "WIND_SPEED-7^1": 0.0,
        # "WIND_SPEED-8^1": 0.0,
        # "WIND_SPEED-9^1": 0.0,
        # "WS_HR-1^1": 0.0,
        # "WS_HR-2^1": 0.0,
        # "WS_HR-3^1": 0.0,
        # "WS_HR-4^1": 0.0,
        # "WS_HR-5^1": 0.0,
        # "WS_HR-6^1": 0.0,
        # "WS_HR-7^1": 0.0,
        # "WS_HR-8^1": 0.0,
        # "WS_HR-9^1": 0.0,
        }

# Parameters for Adam.
# See http://sebastianruder.com/optimizing-gradient-descent/index.html#adam
beta_1 = 0.9
beta_2 = 0.999
epsilon = 0.00000001

is_freezed = False
t = 0
history = {
        "bias_gradient": [0.0],
        "bias_m_t": [0.0],
        "bias_v_t": [0.0],
        "feature_gradients": [{ k: 0.0 for k in feature_config }],
        "feature_m_ts": [{ k: 0.0 for k in feature_config }],
        "feature_v_ts": [{ k: 0.0 for k in feature_config }]
        }
nth_hour = 0
nth_month = 0
loss = 0
while t < num_iterations and not is_freezed:
    t += 1

    model = Model(bias, feature_config)

    bias_gradient = 0.0
    feature_gradients = { k: 0.0 for k in feature_config }

    for j in range():
        feature_values = {
                k: l[nth_month][nth_hour:nth_hour + 10]
                for k, l in training_data.items()
                }
        nth_hour += 1
        if nth_hour == (24 * 20 - 10):
            nth_hour = 0
            nth_month += 1
            if nth_month == 12:
                nth_month = 0
                print("loss:", loss)
                loss = 0
        real_y = feature_values["PM2.5"][9]

        loss += model.calculate_loss_root(feature_values, real_y) ** 2

    for j in range(num_examples):
        feature_values = {
                k: l[nth_month][nth_hour:nth_hour + 10]
                for k, l in training_data.items()
                }
        nth_hour += 1
        if nth_hour == (24 * 20 - 10):
            nth_hour = 0
            nth_month += 1
            if nth_month == 12:
                nth_month = 0
        real_y = feature_values["PM2.5"][9]
        bias_gradient += model.calculate_bias_gradient(feature_values, real_y)
        for k in feature_config:
            feature_gradients[k] += model.calculate_feature_gradient(
                    feature_values, real_y, k)

    bias_m_t = beta_1 * history["bias_m_t"][t - 1] + (
            1 - beta_1) * bias_gradient
    bias_v_t = beta_2 * history["bias_v_t"][t - 1] + (
            1 - beta_2) * (bias_gradient ** 2)
    bias_m_t_hat = bias_m_t / (1 - (beta_1 ** t))
    bias_v_t_hat = bias_v_t / (1 - (beta_2 ** t))
    feature_m_ts = {
            k: beta_1 * history["feature_m_ts"][t - 1][k] + (
                1 - beta_1) * v for k, v in feature_gradients.items()
            }
    feature_v_ts = {
            k: beta_2 * history["feature_v_ts"][t - 1][k] + (
                1 - beta_2) * (v ** 2) for k, v in feature_gradients.items()
            }
    feature_m_t_hats = {
            k: v / (1 - (beta_1 ** t)) for k, v in feature_m_ts.items()
            }
    feature_v_t_hats = {
            k: v / (1 - (beta_2 ** t)) for k, v in feature_v_ts.items()
            }

    bias -= (learning_rate * bias_m_t_hat / (math.sqrt(bias_v_t_hat) + epsilon))
    for k in feature_config:
        feature_config[k] -= (learning_rate * feature_m_t_hats[k] / (
            math.sqrt(feature_v_t_hats[k]) + epsilon))

    history["bias_gradient"].append(bias_gradient);
    history["bias_m_t"].append(bias_m_t);
    history["bias_v_t"].append(bias_v_t);
    history["feature_gradients"].append(feature_gradients);
    history["feature_m_ts"].append(feature_m_ts);
    history["feature_v_ts"].append(feature_v_ts);

    if abs(bias_gradient) < freeze_gradient and all(
            [ abs(v) < freeze_gradient for k, v in feature_gradients.items() ]):
        print(t, bias_gradient)
        is_freezed = True

model = Model(bias, feature_config)
num_validations = num_iterations
delta = 0.0
for i in range(num_validations):
    feature_values = {
            k: l[nth_month][nth_hour:nth_hour + 10]
            for k, l in training_data.items()
            }
    nth_hour += 1
    if nth_hour == (24 * 20 - 10):
        nth_hour = 0
        nth_month += 1
        if nth_month == 12:
            nth_month = 0
    real_y = feature_values["PM2.5"][9]
    delta += (abs(real_y - model.calculate_y(feature_values)) ** 2)
print("Final RMSE:", math.sqrt(delta / num_validations))

testing_data = {}
with open(testing_file_name, "r", encoding = "big5") as testing_file:
    testing_csv = csv.reader(testing_file)
    for row in testing_csv:
        if row:
            i = row[0]
            index_name = row[1]
            if i not in testing_data:
                testing_data[i] = {}
            testing_data[i][index_name] = [ (
                float(s)) if s != "NR" else 0.0 for s in row[2:] ]

output_string = "id,value\n"
for k, feature_values in testing_data.items():
    output_string += (k + "," + str(model.calculate_y(feature_values)) + "\n")

with open(output_file_name, "w") as f:
    f.write(output_string)

#!/usr/bin/env python

import csv
import sys

raw_training_file_name = sys.argv[1]

num_training_data = 0
num_correct_y = 0
with open(raw_training_file_name, "r") as raw_training_file:
    raw_training_csv = csv.reader(raw_training_file)
    for row in raw_training_csv:
        num_training_data += 1
        edu_num = int(row[4].strip())
        race = row[8].strip()
        sex = row[9].strip()
        y_hat = 1 if row[14].strip() == ">50k" else 0
        y = 0
        if edu_num >= 13 and race == "White" and sex == "Male":
            y = 1
        if y == y_hat:
            num_correct_y += 1

accuracy = num_correct_y / float(num_training_data)

print("")
print("========== Summary ==========")
print("ACCURACY: %f%%" % (accuracy * 100))
print("")

raw_testing_file_name = sys.argv[2]
testing_y = []
nth_row = 0
with open(raw_testing_file_name, "r") as raw_testing_file:
    raw_testing_csv = csv.reader(raw_testing_file)
    for row in raw_testing_csv:
        nth_row += 1
        if nth_row != 1:
            edu_num = int(row[4].strip())
            race = row[8].strip()
            sex = row[9].strip()
            y = 0
            if edu_num >= 13 and race == "White" and sex == "Male":
                y = 1
            testing_y.append(y)

output_file_name = sys.argv[6]
with open(output_file_name, "w") as output_file:
    output_file.write("id,label\n")
    i = 0
    for y in testing_y:
        i += 1
        output_file.write("%s,%s\n" %(str(i), str(y)))

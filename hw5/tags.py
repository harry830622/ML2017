#!/usr/bin/env python3

from extract import extract

import numpy as np

import sys
import pickle

training_file_name = sys.argv[1]
testing_file_name = sys.argv[2]
label_mapping_file_name = sys.argv[3]
output_file_name = sys.argv[4]

training_x, training_y, tokenizer, testing_x, classes = extract(
    training_file_name, testing_file_name)

with open(label_mapping_file_name, "rb") as label_mapping_file:
    classes = pickle.load(label_mapping_file)
classes = np.array(classes)

tags_distribution = {
    classes[i]: n
    for i, n in enumerate(np.sum(training_y, axis=0))
}

with open(output_file_name, "w") as output_file:
    output_file.write("tag,number\n")
    for k, v in tags_distribution.items():
        output_file.write("{},{:d}\n".format(k, v))

#!/usr/bin/env python

import sys
import numpy as np

def file_to_matrix(file_name):
    matrix_list = []
    with open(file_name, "r") as input_file:
        for line in input_file:
            row_list = [int(token) for token in line.strip("\n").split(",")]
            matrix_list.append(row_list)
    return np.matrix(matrix_list)

matrix_a = file_to_matrix(sys.argv[1])
matrix_b = file_to_matrix(sys.argv[2])
result_matrix = matrix_a.dot(matrix_b)
result = [n for row_list in result_matrix.tolist() for n in row_list]
result.sort()

with open("ans_one.txt", "w") as output_file:
    output_file.write("\n".join([str(n) for n in result]))

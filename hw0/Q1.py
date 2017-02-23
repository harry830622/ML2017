#!/usr/bin/env python

import sys
import numpy as np

matrix_a = []
with open(sys.argv[1], "r") as matrix_a_file:
    for line in matrix_a_file:
        row = [int(s) for s in line.strip("\n").split(",")]
        matrix_a.append(row)

matrix_b = []
with open(sys.argv[2], "r") as matrix_b_file:
    for line in matrix_b_file:
        row = [int(s) for s in line.strip("\n").split(",")]
        matrix_b.append(row)

result_matrix = np.matrix(matrix_a).dot(np.matrix(matrix_b))
result = [s for sublist in result_matrix.tolist() for s in sublist]
result.sort()

with open("ans_one.txt", "w") as output_file:
    output_file.write("\n".join([str(n) for n in result]))

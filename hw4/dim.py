#!/usr/bin/env python3

import numpy as np
import sys
import os

data = np.load(sys.argv[1])
std = []
for key in data.keys():
    std.append(np.std(data[key]))
std = np.array(std)
std2 = std**2
min = np.min(std2)
step = (np.max(std2) - min)/60
out = np.log((std2-min)/step + 1.)

output_file_name = sys.argv[2]
with open(output_file_name, "w") as output_file:
    output_file.write("SetId,LogDim\n")
    for i, n in enumerate(out):
        output_file.write("%d,%f\n" % (i, n))

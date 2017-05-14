#!/usr/bin/env python3

import numpy as np

from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues

import sys
import pickle

np.random.seed(19940622)

with open("model", "rb") as model_file:
    svr = pickle.load(model_file)

test_data = np.load(sys.argv[1])
test_X = []
for i in range(200):
    data = test_data[str(i)]
    vs = get_eigenvalues(data)
    test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)

with open(sys.argv[2], "w") as f:
    f.write("SetId,LogDim\n")
    for i, d in enumerate(pred_y):
        f.write("{:d},{:f}\n".format(i, np.log(np.round(d))))

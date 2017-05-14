#!/usr/bin/env python3

from gen import get_eigenvalues

import numpy as np

from sklearn.svm import LinearSVR as SVR

import sys
import pickle

npzfile = np.load(sys.argv[1])
X = npzfile["X"]
y = npzfile["y"]

svr = SVR(C=50)
svr.fit(X, y)

with open("model.p", "wb") as model_file:
    pickle.dump(svr, model_file)

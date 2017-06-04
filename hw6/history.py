#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys
import pickle

colors = ["r", "g", "b", "c", "m", "y", "k"]

ds = [18, 50, 100, 120, 200, 500, 1000]

historys = []
for d in ds:
    with open("history_{:d}d_0.p".format(d), "rb") as f:
        historys.append(pickle.load(f))

historys = [(np.sqrt(np.min(h["val_mean_squared_error"])), d, h)
            for d, h in zip(ds, historys)]
historys = sorted(historys, key=lambda t: t[0])

plt.figure(figsize=(12, 9))

legends = []
for (min_valid, d, h), c in zip(historys, colors):
    print("{:d}: {:.3f}".format(d, min_valid))
    plt.plot(h["mean_squared_error"], "{}--".format(c))
    plt.plot(h["val_mean_squared_error"], "{}-".format(c))
    legends += ["{:d}d_train".format(d), "{:d}d_valid".format(d)]

plt.title("Comparison of various latent dimensions")
plt.ylabel("MSE")
plt.xlabel("# of epochs")
plt.legend(legends, loc="lower right")

plt.savefig("history_dimension.png")
plt.close()

with open("history_all_0.p", "rb") as f:
    history = pickle.load(f)

with open("history_nonorm_0.p", "rb") as f:
    history_nonorm = pickle.load(f)

plt.figure(figsize=(12, 9))

plt.plot(history["mean_squared_error"], "r--")
plt.plot(history["val_mean_squared_error"], "r-")
plt.plot(history_nonorm["mean_squared_error"], "b--")
plt.plot(history_nonorm["val_mean_squared_error"], "b-")

plt.title("Normalized vs. Non-normalized")
plt.ylabel("MSE")
plt.xlabel("# of epochs")
plt.legend(
    ["norm_train", "norm_valid", "nonorm_train", "nonorm_valid"],
    loc="lower right")

plt.savefig("history_nonorm.png")
plt.close()

with open("history_all_0.p", "rb") as f:
    history = pickle.load(f)

with open("history_nobias_0.p", "rb") as f:
    history_nobias = pickle.load(f)

plt.figure(figsize=(12, 9))

plt.plot(history["mean_squared_error"], "r--")
plt.plot(history["val_mean_squared_error"], "r-")
plt.plot(history_nobias["mean_squared_error"], "b--")
plt.plot(history_nobias["val_mean_squared_error"], "b-")

plt.title("Biased vs. Non-biased")
plt.ylabel("MSE")
plt.xlabel("# of epochs")
plt.legend(
    ["bias_train", "bias_valid", "nobias_train", "nobias_valid"],
    loc="lower right")

plt.savefig("history_nobias.png")
plt.close()

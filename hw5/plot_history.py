#!/usr/bin/env python3

import matplotlib.pyplot as plt

import sys
import pickle

history_file_name = sys.argv[1]
suffix = sys.argv[2]

with open(history_file_name, "rb") as history_file:
    history = pickle.load(history_file)

plt.plot(history['f1_score'])
plt.plot(history['val_f1_score'])
plt.title('Model F1 Score')
plt.ylabel('F1 Score')
plt.xlabel('# of epochs')
plt.legend(['train', 'valid'], loc='lower right')
plt.savefig("history_{}.png".format(suffix))
plt.close()

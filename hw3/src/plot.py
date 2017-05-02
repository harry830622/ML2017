#!/usr/bin/env python3

import pickle
import sys

from keras.models import model_from_json
from keras.utils import plot_model

model_file_name = sys.argv[1]

with open(model_file_name, "rb") as model_file:
    m = pickle.load(model_file)
    model = model_from_json(m["config"])
    model.set_weights(m["weights"])
    history = m["history"]

model.summary()

plot_model(model, "structure.png")

print(history)

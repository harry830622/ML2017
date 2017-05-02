#!/usr/bin/env bash

training_x="./training_x.p"
training_y="./training_y.p"

python3 "./src/cnn.py" "$training_x" "$training_y" "model.p"

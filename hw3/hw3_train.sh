#!/usr/bin/env bash

python3 "./src/cnn.py" "$1" "model.h5"
python3 "./src/semi.py" "$1" "test.csv" "model.h5"

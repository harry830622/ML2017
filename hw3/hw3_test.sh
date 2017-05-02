#!/usr/bin/env bash

testing_x="./testing_x.p"

python3 "./src/predict.py" "$testing_x" "$2" "model.p"

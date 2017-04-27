#!/usr/bin/env bash

training_x="./training_x.p"
training_y="./training_y.p"

if [ ! -f "$training_x" ]; then
  python3 "./src/extract_training_data.py" "$1" "$training_x" "$training_y"
fi

python3 "./src/train.py" "$training_x" "$training_y" "model.p"

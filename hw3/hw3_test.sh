#!/usr/bin/env bash

testing_x="./testing_x.p"

if [ ! -f "$testing_x" ]; then
  python3 "./src/extract_testing_data.py" "$1" "$testing_x"
fi

python3 "./src/test.py" "$testing_x" "$2" "model.p"

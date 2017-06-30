#!/usr/bin/env bash

if [ ! -d cleaned_data ]; then
  mkdir cleaned_data
fi

python3.6 ./src/preprocess.py "$1" "$2" cleaned_data

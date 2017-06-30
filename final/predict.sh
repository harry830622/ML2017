#!/usr/bin/env bash

if [ ! -f model_rf_0.p ]; then
  wget http://eda.ee.ntu.edu.tw/~yhchang/ml2017_final_model/model_rf_0.p
fi

if [ ! -f model_xgb_0 ]; then
  wget http://eda.ee.ntu.edu.tw/~yhchang/ml2017_final_model/model_xgb_0
  wget http://eda.ee.ntu.edu.tw/~yhchang/ml2017_final_model/model_xgb_1
  wget http://eda.ee.ntu.edu.tw/~yhchang/ml2017_final_model/model_xgb_2
  wget http://eda.ee.ntu.edu.tw/~yhchang/ml2017_final_model/model_xgb_3
  wget http://eda.ee.ntu.edu.tw/~yhchang/ml2017_final_model/model_xgb_4
fi

python3.6 ./src/predict.py ./cleaned_data/cleaned_x_test.p "$1" .

# -*- coding: UTF-8 -*-
"""
Author：LIAO XUECHUN
Data：2023/1/12
Describe：multivariate_single_step
"""
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import ms_args_parser
from util import train, test_m, load_data_m,forecast_m
import pandas as pd

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/models/multivariate_single_step.pkl'


if __name__ == '__main__':
    args = ms_args_parser()
    flag = 'ms'
    Dtr, Val, Dte,scaler = load_data_m(args, flag)
    train(args, Dtr, Val, LSTM_PATH)
    mape,test=test_m(args, Dte, LSTM_PATH,scaler)
    forecast=forecast_m(args, LSTM_PATH,scaler)
    forecast_result=pd.concat([test,forecast])
    def result():
        if args.method =='yuce':
            result_print=forecast_result
        else:
            result_print = mape
        return result_print
    result()


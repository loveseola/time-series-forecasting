# -*- coding: utf-8 -*-
"""
Author：LIAO XUECHUN
Data：2023/1/12
Describe：univariate_single_step
"""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from util import train, test, load_data_u,forecast,forecast_m
from args import us_args_parser
from data_process import setup_seed
import pandas as pd

setup_seed(20)
path = os.path.abspath(os.path.dirname(os.getcwd()))#当前大文件夹目录
LSTM_PATH = path + '/models/univariate_single_step.pkl'


if __name__ == '__main__':
    args = us_args_parser()
    flag = 'us'
    Dtr, Val, Dte, m, n = load_data_u(args, flag)
    train(args, Dtr, Val, LSTM_PATH)
    mape,test=test(args, Dte, LSTM_PATH, m, n)
    forecast_result=forecast(args, LSTM_PATH)
    forecast_result=pd.concat([test,forecast_result])
    def result():
        if args.method =='yuce':
            result_print=forecast_result
        else:
            result_print = mape
        return result_print
    result()

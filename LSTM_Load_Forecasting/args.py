# -*- coding: UTF-8 -*-
import argparse, torch

def mm_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=3, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=30, help='seq len')
    parser.add_argument('--output_size', type=int, default=3, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=4, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--yucestart',type=int,default=5,help='forecasting start')
    parser.add_argument('--yuceend',type=int,default=7,help='forecasting end')
    parser.add_argument('--train_split',type=int,default=0.6,help='train split')
    parser.add_argument('--test_split',type=int,default=0.8,help='test split ')
    parser.add_argument('--predict_df',type=str,default='F:/E/BaiduSyncdisk/lxc/tsf/tsf_finally/LSTM_Load_Forecasting/data/predict_df.xlsx',help='predict df ')
    parser.add_argument('--df',type=str,default='F:/E/BaiduSyncdisk/lxc/tsf/tsf_finally/LSTM_Load_Forecasting/data/shengzhu.xlsx',help='df ')
    parser.add_argument('--day', type=int, default=30,
                        help='yuce day ')
    parser.add_argument('--method', type=str, default='yuce',
                        help='is predict of mape ')
    args = parser.parse_args()
    return args


def ms_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='input dimension')
    parser.add_argument('--input_size', type=int, default=3, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=30, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--yucestart',type=int,default=5,help='forecasting start')
    parser.add_argument('--yuceend',type=int,default=7,help='forecasting end')
    parser.add_argument('--train_split',type=int,default=0.6,help='train split')
    parser.add_argument('--test_split',type=int,default=0.8,help='test split ')
    parser.add_argument('--predict_df',type=str,default='F:/E/BaiduSyncdisk/lxc/tsf/tsf_finally/LSTM_Load_Forecasting/data/predict_df.xlsx',help='predict df ')
    parser.add_argument('--df',type=str,default='F:/E/BaiduSyncdisk/lxc/tsf/tsf_finally/LSTM_Load_Forecasting/data/shengzhu.xlsx',help='df ')
    parser.add_argument('--day', type=int, default=30,
                        help='yuce day ')
    parser.add_argument('--method', type=str, default='yuce',
                        help='is predict of mape ')
    args = parser.parse_args()
    return args


def us_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='diedai num')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=30, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')#是否是双向LSTM
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--yucel',type=int,default=7,help='forecasting ')
    parser.add_argument('--train_split',type=int,default=0.6,help='train split')
    parser.add_argument('--test_split',type=int,default=0.8,help='test split ')
    parser.add_argument('--df',type=str,default='F:/E/BaiduSyncdisk/lxc/tsf/tsf_finally/LSTM_Load_Forecasting/data/shengzhu.xlsx',help='df ')
    parser.add_argument('--day', type=int, default=30,
                        help='yuce day ')
    parser.add_argument('--method', type=str, default='yuce',
                        help='is predict of mape ')
    args = parser.parse_args()
    return args
# okay decompiling args.cpython-37.pyc

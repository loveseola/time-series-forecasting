# -*- coding: UTF-8 -*-
import copy
import os
import sys

import pandas as pd

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from itertools import chain
import torch
from scipy.interpolate import make_interp_spline
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import LSTM, BiLSTM
from data_process import nn_seq_us, nn_seq_ms, nn_seq_mm, device, get_mape, setup_seed, MyDataset,load_datas,data_gy,insertscaler
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from datetime import timedelta
import datetime
from sklearn.preprocessing import MinMaxScaler
setup_seed(20)

def load_data_m(args, flag):
    if flag == 'ms':
        Dtr, Val, Dte,scaler = nn_seq_ms(args=args,seq_len=(args.seq_len), B=(args.batch_size),train_split=args.train_split,test_split=args.test_split)
    else:
        Dtr, Val, Dte,scaler = nn_seq_mm(args=args,seq_len=(args.seq_len), B=(args.batch_size), num=(args.output_size),train_split=args.train_split,test_split=args.test_split)
    return (
     Dtr, Val, Dte,scaler)
def load_data_u(args, flag):
    if flag == 'us':
        Dtr, Val, Dte, m, n = nn_seq_us(args=args,seq_len=(args.seq_len), B=(args.batch_size),train_split=args.train_split,test_split=args.test_split)
    return (
     Dtr, Val, Dte, m, n)

def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)# 定义误差函数
    val_loss = []
    for seq, label in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam((model.parameters()), lr=(args.lr), weight_decay=(args.weight_decay))
    else:
        optimizer = torch.optim.SGD((model.parameters()), lr=(args.lr), momentum=0.9,
          weight_decay=(args.weight_decay))
    scheduler = StepLR(optimizer, step_size=(args.step_size), gamma=(args.gamma))
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for seq, label in Dtr:
            seq = seq.to(device)#因为前期处理进行了torch.FloatTensor,这里直接to(device)。shape为（batch_size,seq_len,int(args.yuceend)+1-int(args.yucestart))
            label = label.to(device)#shape为（batch_size,int(args.yuceend)+1-int(args.yucestart))
            optimizer.zero_grad()
            y_pred = model(seq)#前向传播
            loss = loss_function(y_pred, label)#计算损失
            loss.backward()# 误差反向传播
            optimizer.step()# 梯度更新
            train_loss.append(loss.item())  # 损失记录
        scheduler.step()
        val_loss = get_val_loss(args, model, Val)
        if epoch + 1 >= min_epochs:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)

def test_m(args, Dte, path,scaler,flag):
    '''
    with torch.no_grad() 是创建了一张图，所以在这张图中，所有的计算都不会进入梯度计算；detach()将一个变量从图中剥离出来进而使得此变量的计算不会进入到梯度计算中
    :param args: 参数
    :param Dte: 测试集
    :param path: 加载模型
    :param scaler: 标准化
    :param flag: 模型代称
    :return: 评估结果和测试集原始预测
    '''
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('testing...')
    for seq, target in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))#target.data.tolist()tensor数据转化成列表，再迭代成列表[[1,2],[3,4]]=>[1,2,3,4]
        y.extend(target)#数据扩展，在已存在的列表中添加新的列表内容
        seq = seq.to(device)#shape是(batch_size,seq_len,dim）
        #print(seq.shape)#torch.Size([30, 30, 1])...torch.Size([1, 30, 1])，一共361个
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
    y, pred = np.array(y), np.array(pred)#注意这里的y和pred是列表
    #逆变化
    y=insertscaler(y,args)
    pred=insertscaler(pred,args)
    if flag=='ms':
        pred_result=pd.DataFrame({'original':y,'pred':pred},index = load_datas(args.df)[int(len(load_datas(args.df)) * args.test_split)+int(args.seq_len):len(load_datas(args.df))].index)
    if flag=='mm':
        pred_result = pd.DataFrame({'original': y, 'pred': pred}, index=load_datas(args.df)[int(len(
            load_datas(args.df)) * args.test_split) + int(args.seq_len):int(len(
            load_datas(args.df)) * args.test_split) + int(args.seq_len)+len(y)].index)
    # 评估
    print('mape:', get_mape(y, pred))
    return pd.DataFrame([get_mape(y, pred)],columns=['mape']),pred_result
def test(args, Dte, path, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('testing...')
    for seq, target in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)#shape是(batch_size,seq_len,dim）
        #print(seq.shape)#torch.Size([30, 30, 1])...torch.Size([1, 30, 1])，一共361个
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
    y, pred = np.array(y), np.array(pred)
    #归一化处理
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    pred_result = pd.DataFrame({'original': y, 'pred': pred}, index=load_datas(args.df)[int(len(load_datas(args.df)) * args.test_split) + int(args.seq_len):len(load_datas(args.df))].index)
    # 评估
    print('mape:', get_mape(y, pred))
    return pd.DataFrame([get_mape(y, pred)],columns=['mape']),pred_result
    #x = [i for i in range(1, 151)]
    #x_smooth = np.linspace(np.min(x), np.max(x), 900)#在最大值和最小值之间生成900个数
    #y_smooth = make_interp_spline(x, y[150:300])(x_smooth)#使之平滑
    #plt.plot(y)
    #plt.plot(pred)
    #plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')
    #y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    #plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    #plt.grid(axis='y')
    #plt.legend()
    #plt.show()
def forecast(args, path):
    print('forecasting')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    model.load_state_dict(torch.load(path)['models'])
    #原始值
    df_day_past=load_datas(args.df)[-args.seq_len:]
    df_day_past=df_day_past.iloc[:,int(args.yucel)]#shape为（len(df_day_past),)可以np.array(df_day_past.iloc[:,int(args.yucel)]).reshape(-1,1)变为二维（len(df_day_past),1)
    past_array = data_gy(df_day_past).tolist()  # 归一
    #未来值
    new_array=[0]*args.day
    past_array.extend(new_array)
    a1=datetime.datetime.strptime(str(df_day_past.index[-1]),'%Y-%m-%d')+timedelta(days=1)
    a2=a1+timedelta(days=args.day-1)

    for x in range(0,len(past_array)-args.seq_len):
        data_x=[]
        for j in range(x,x+args.seq_len):
             i=[past_array[j]]
             data_x.append(i)
        data_x = torch.FloatTensor([data_x])#增加维度
        #data_x = MyDataset(data_x)
        #data_x = DataLoader(dataset=data_x, batch_size=args.batch_size, shuffle=False, num_workers=0)
        #data_x = [x for x in iter(data_x)][0]
        data_x = data_x.to(device)
        with torch.no_grad():
            x_pred = model(data_x)
            x_pred = list(chain.from_iterable(x_pred.data.tolist()))
            past_array[x+args.seq_len]=x_pred[0]
    yucel = [round(x * (np.max(df_day_past) - np.min(df_day_past)) + np.min(df_day_past),2) for x in past_array]
    yucel_data=pd.DataFrame({'pred':yucel[args.seq_len:]},index=pd.date_range(a1,a2))
    return yucel_data
def forecast_m(args, path,scaler):
    print('forecasting')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=(args.batch_size)).to(device)
    model.load_state_dict(torch.load(path)['models'])
    #原始值
    if args.method =='ms':
        df_day_past=load_datas(args.df)[-args.seq_len:]
    else:
        df_day_past = load_datas(args.df)[-(args.seq_len+args.day-args.output_size):]
    df_day_past=df_day_past.iloc[:,int(args.yucestart):int(args.yuceend)+1]#shape为(len(df_day_past),int(args.yuceend)-int(args.yucestart))
    df_day_past_array=scaler.transform(df_day_past)
    #未来值
    a0=datetime.datetime.strptime(str(df_day_past.index[0]),'%Y-%m-%d')
    a1=datetime.datetime.strptime(str(df_day_past.index[-1]),'%Y-%m-%d')+timedelta(days=1)
    a2=a1+timedelta(days=args.day-1)
    df_day_future=load_datas(args.predict_df)
    df_day_future.index=pd.date_range(a1,a2)
    df_day_future[str(df_day_past.columns[-1])]=0
    df_day_future.columns = df_day_past.columns
    df_day_future_array=scaler.transform(df_day_future)#变成了array
    new_array=pd.DataFrame(df_day_future_array)
    new_array.iloc[:,int(args.yuceend)-int(args.yucestart)]=np.nan#缩放后将未来预测列替换成“nan"
    full_df=pd.concat([pd.DataFrame(df_day_past_array),new_array])
    full_df_array=full_df.values#对于dataframe需要先取values,shape为(60,int(args.yuceend)+1-int(args.yucestart))
    if args.method =='ms':
        for x in range(0,args.day):
            data_x=[]
            data_x.append(full_df_array[x:x+args.seq_len,0:int(args.input_size)])
            data_x = torch.FloatTensor(data_x)#shape为(1,args.seq_len,int(args.yuceend)+1-int(args.yucestart))
            data_x = data_x.to(device)
            with torch.no_grad():
                x_pred = model(data_x)
                x_pred = list(chain.from_iterable(x_pred.data.tolist()))
                full_df.iloc[x+args.seq_len,int(args.yuceend)-int(args.yucestart)]=x_pred[0]
    else:
        for x in range(0,args.day,args.output_size):
            data_x=[]
            data_x.append(full_df_array[x:x+args.seq_len,0:int(args.input_size)])
            data_x = torch.FloatTensor(data_x)#shape为(1,args.seq_len,int(args.yuceend)+1-int(args.yucestart))
            data_x = data_x.to(device)
            with torch.no_grad():
                x_pred = model(data_x)
                x_pred = list(chain.from_iterable(x_pred.data.tolist()))
                for y in range(0,args.output_size):
                    full_df.iloc[x+(args.seq_len+args.day-args.output_size)+y,int(args.yuceend)-int(args.yucestart)]=x_pred[y]
    #逆变化
    yucel=np.array(full_df.iloc[:,int(args.yuceend)-int(args.yucestart)]).reshape(-1,1)#改变shape为(len(full_df),1)
    yucel=insertscaler(yucel,args)
    yucel_data=pd.DataFrame({'pred':yucel[(args.seq_len+args.day-args.output_size):]},index=pd.date_range(a1,a2))
    return yucel_data



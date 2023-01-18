# -*- coding:utf-8
"""
author:Liao Xuechun
date:2022-07-13
describe:keras_model time series forecasting
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from datetime import timedelta
import datetime
import openpyxl

data=pd.read_excel('广东地区企业报价.xlsx')
data.columns=['date','ws','zb']
ycls = 1
ycle = 2
ycl = 0
cut = 9
day = 30
srn = 16
zjn =6
dropout=0.2
#dataset
class keras:
    def __init__(self,ycls,ycle,ycl,cut,day,datapath):
        self.ycls=int(ycls)#预测开始列
        self.ycle=int(ycle)#预测结束列
        self.ycl=int(ycl)#目标预测列差
        self.cut=int(cut)#训练集比例划分，取1-10之间的整数
        self.day=int(day)#预测下一个目标值将在过去查看的步骤数
        self.datapath=pd.DataFrame(datapath)
    def kerasdata(self):
        global data
        data=self.datapath
        data['date']=pd.to_datetime(data['date'])
        data=data.set_index('date')
        global data_
        data_=pd.DataFrame(data.iloc[:,int(self.ycls):int(self.ycle)])
        data_=data_.astype('float')
        global dataset
        dataset = data_.values#单变量必须要pd.Dataframe后变成二维
        global cut_
        cut_=len(dataset)*int(self.cut)//10
        train,valid=dataset[:cut_],dataset[cut_:]
        global scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        global scaled_data,scaled_train,scaled_valid
        scaled_data,scaled_train,scaled_valid= scaler.fit_transform(dataset),scaler.fit_transform(train),scaler.fit_transform(valid)
        return scaled_data,scaled_train,scaled_valid
    def creatXY(self,kerasdata):
        traindataX,traindataY,testdataX,testdataY,datadataX,datadataY=[],[],[],[],[],[]
        for i in range(self.day,len(keras.kerasdata(self)[1])):
            traindataX.append(keras.kerasdata(self)[1][i-self.day:i,0:keras.kerasdata(self)[1].shape[1]])
            traindataY.append(keras.kerasdata(self)[1][i,int(self.ycl)])
        trainX,trainY=np.array(traindataX),np.array(traindataY)
        for i in range(self.day,len(keras.kerasdata(self)[2])):
            testdataX.append(keras.kerasdata(self)[2][i-self.day:i,0:keras.kerasdata(self)[2].shape[1]])
            testdataY.append(keras.kerasdata(self)[2][i,int(self.ycl)])
        testX,testY=np.array(testdataX),np.array(testdataY)
        for i in range(self.day,len(keras.kerasdata(self)[0])):
            datadataX.append(keras.kerasdata(self)[0][i-self.day:i,0:keras.kerasdata(self)[0].shape[1]])
            datadataY.append(keras.kerasdata(self)[0][i,int(self.ycl)])
        dataX,dataY=np.array(datadataX),np.array(datadataY)
        trainX,trainY,testX,testY,dataX,dataY=trainX.reshape(len(keras.kerasdata(self)[1])-int(self.day),int(self.day)),trainY.reshape(len(keras.kerasdata(self)[1])-int(self.day),1),testX.reshape(len(keras.kerasdata(self)[2])-int(self.day),int(self.day)),testY.reshape(len(keras.kerasdata(self)[2])-int(self.day),1),dataX.reshape(len(keras.kerasdata(self)[0])-int(self.day),int(self.day)),dataY.reshape(len(keras.kerasdata(self)[0])-int(self.day),1)
        return trainX,trainY,testX,testY,dataX,dataY
c=keras(ycls,ycle,ycl,cut,day,data)
c1=c.kerasdata()
c2=c.creatXY(c1)
#net
def build_model_bp(optimizer):
        grid_model = Sequential()
        #第一层
        grid_model.add(Dense(int(srn),input_dim=int(day),kernel_initializer='uniform')) #输入层，Dense表示BP层
        grid_model.add(Activation('relu'))  #添加激活函数
        grid_model.add(Dense(int(zjn),kernel_initializer='uniform')) #中间层
        grid_model.add(Activation('sigmoid'))  #添加激活函数
        grid_model.add(Dense(1))  #输出层
        grid_model.compile(loss='mean_squared_error', optimizer=optimizer) #编译模型
        return grid_model
grid_model_bp = KerasRegressor(build_fn=build_model_bp,verbose=1)
parameters_bp = {'batch_size':[16,20],'epochs':[8,10],'optimizer':['adam','Adadelta'] }
grid_search_bp  = GridSearchCV(estimator = grid_model_bp,
                            param_grid = parameters_bp)
grid_search_bp = grid_search_bp.fit(c2[0],c2[1],validation_data=(c2[2],c2[3]))#防止过拟合
my_model_bp=grid_search_bp.best_estimator_.model
def build_model_lstm(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(int(srn),return_sequences=True,input_shape=(int(day),int(ycle)-int(ycls))))#输入层
    grid_model.add(LSTM(int(zjn)))#中间层
    grid_model.add(Dropout(float(dropout)))
    grid_model.add(Dense(1))
    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model
grid_model_lstm = KerasRegressor(build_fn=build_model_lstm,verbose=1)
parameters_lstm = {'batch_size':[16,20],'epochs':[8,10],'optimizer':['adam','Adadelta'] }
grid_search_lstm  = GridSearchCV(estimator = grid_model_lstm,
                            param_grid = parameters_lstm,
                            cv = 2)
grid_search_lstm = grid_search_lstm.fit(c2[0],c2[1],validation_data=(c2[2],c2[3]))#防止过拟合
my_model_lstm=grid_search_lstm.best_estimator_.model
#predict and estimate
def bpkeras_pe(data_,model,model_name,model_names):
	prediction_=model.predict(c2[2])#test
	pred=scaler.inverse_transform(prediction_)[:,int(ycl)]
	original=scaler.inverse_transform(c2[3])
	df__days_past=data_.iloc[-int(day):,:]
	a1 = pd.to_datetime(data_.index[-1])+timedelta(days=1)
	a2=a1+timedelta(days=int(day)-1)
	df__days_future=pd.DataFrame(np.random.randn(int(day),int(ycle)-int(ycls)),index=pd.date_range(a1,a2),columns=data_.columns)
	df__days_future.iloc[:,int(ycl)]=0
	old_scaled_array=scaler.transform(df__days_past)
	new_scaled_array=scaler.transform(df__days_future)#缩放前变成0
	new_scaled_df=pd.DataFrame(new_scaled_array)
	new_scaled_df.iloc[:,int(ycl)]=np.nan
	full_df=pd.concat([pd.DataFrame(old_scaled_array),new_scaled_df]).reset_index().drop(["index"],axis=1)#拼在一起
	full_df_scaled_array=full_df.values
	all_data=[]
	time_step=int(day)
	for i in range(time_step,len(full_df_scaled_array)):
		data_x=[]
		data_x.append(full_df_scaled_array[i-time_step:i,0:full_df_scaled_array.shape[1]])
		data_x0=np.array(data_x).reshape(1,int(day))
		prediction=model.predict(data_x0)
		all_data.append(prediction)
		full_df.iloc[i,0]=prediction
	new_array=np.array(all_data)#一维
	new_array=new_array.reshape(-1,1)#变成两维
	y_pred_future__days = scaler.inverse_transform(new_array)[:,0]#变回去
	y_pred_future=pd.DataFrame(y_pred_future__days,index=pd.date_range(a1,a2),columns=[model_name])
    #整理成预测和未来的
	y_pred=pd.DataFrame(original,pred).reset_index()
	y_pred.index=data.index[cut_+(int(day)):]
	y_pred=y_pred.astype('float')
	y_pred.columns=[model_name,'original']
	y_pred=y_pred.append(y_pred_future,sort=True)
	y_pred=y_pred.apply(lambda x:round(x,2))
	rmse = np.sqrt(np.mean(np.power(y_pred["original"][:-int(day)] - y_pred[model_name][:-int(day)],2)))
	mae=mean_absolute_error(y_pred["original"][:-int(day)],y_pred[model_name][:-int(day)])
	mse=mean_squared_error(y_pred["original"][:-int(day)],y_pred[model_name][:-int(day)])
	r2=r2_score(y_pred["original"][:-int(day)],y_pred[model_name][:-int(day)])
    mape=mean_absolute_percentage_error(y_pred["original"][:-int(day)],y_pred[model_name][:-int(day)])
	model_estimate=pd.DataFrame({model_names:[rmse,mae,mse,r2,,mape]},index=['rmse','mae','mse','r2','mape'])
	return y_pred,model_estimate
result1=bpkeras_pe(data_,my_model_bp,'bp_pred','kerasbp')

def lstmkeras_pe(data_,model,model_name,model_names):
    prediction=model.predict(c2[2])
    prediction_copies_array = np.repeat(prediction,dataset.shape[1], axis=-1)
    pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),dataset.shape[1])))[:,int(ycl)]#目标
    original_copies_array = np.repeat(c2[3],dataset.shape[1], axis=-1)
    original=scaler.inverse_transform(np.reshape(original_copies_array,(len(c2[3]),dataset.shape[1])))[:,int(ycl)]#第一列
    df__days_past=data_.iloc[-int(day):,:]
    a1 = data_.index[-1]+timedelta(days=1)
    a2=a1+timedelta(days=int(day)-1)
    df__days_future=pd.DataFrame(np.random.randn(int(day),int(ycle)-int(ycls)),index=pd.date_range(a1,a2),columns=data_.columns)
    df__days_future.iloc[:,int(ycl)]=0
    old_scaled_array=scaler.transform(df__days_past)
    new_scaled_array=scaler.transform(df__days_future)#缩放前变成0
    new_scaled_df=pd.DataFrame(new_scaled_array)
    new_scaled_df.iloc[:,0]=np.nan
    full_df=pd.concat([pd.DataFrame(old_scaled_array),new_scaled_df]).reset_index().drop(["index"],axis=1)#拼在一起
    full_df_scaled_array=full_df.values
    all_data=[]
    time_step=int(day)
    for i in range(time_step,len(full_df_scaled_array)):
        data_x=[]
        data_x.append(full_df_scaled_array[i-time_step:i,0:full_df_scaled_array.shape[1]])
        data_x=np.array(data_x)
        prediction=model.predict(data_x)
        all_data.append(prediction)
        full_df.iloc[i,0]=prediction
    new_array=np.array(all_data)#一维
    new_array=new_array.reshape(-1,1)#变成两维
    prediction_copies_array = np.repeat(new_array,dataset.shape[1], axis=-1)
    y_pred_future__days = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(new_array),dataset.shape[1])))[:,int(ycl)]#变回去
    y_pred_future=pd.DataFrame(y_pred_future__days,index=pd.date_range(a1,a2),columns=[model_name])
    #整理成预测和未来的
    y_pred=pd.DataFrame(original,pred).reset_index()
    y_pred.index=data.index[cut_+(int(day)):]
    y_pred.columns=[model_name,'original']
    y_pred=y_pred.append(y_pred_future,sort=True)
    result=y_pred
    y_pred=y_pred.apply(lambda x:round(x,2))
    rmse = np.sqrt(np.mean(np.power(y_pred["original"][:-int(day)] - y_pred[model_name][:-int(day)],2)))
    mae=mean_absolute_error(y_pred["original"][:-int(day)],y_pred[model_name][:-int(day)])
    mse=mean_squared_error(y_pred["original"][:-int(day)],y_pred[model_name][:-int(day)])
    r2=r2_score(y_pred["original"][:-int(day)],y_pred[model_name][:-int(day)])
    mape=mean_absolute_percentage_error(y_pred["original"][:-int(day)],y_pred[model_name][:-int(day)])
    model_estimate=pd.DataFrame({model_names:[rmse,mae,mse,r2,mape]},index=['rmse','mae','mse','r2','mape'])
    return y_pred,model_estimate
result2=lstmkeras_pe(data_,my_model_lstm,'lstm_pred','kreaslstm')
# -*- coding:utf-8
"""
author:Liao Xuechun
date:2022-07-12
describe:prophet time series forecasting
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from fbprophet import Prophet
from pandas import read_excel
import holidays
import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data=pd.read_excel('广东地区企业报价.xlsx')
data.columns=['date','ws','zb']
day=30
ycl='ws'
#数据输入
class fbpredict:
    def __init__(self,day,ycl,datapath):
        self.day=day#预测天数
        self.ycl=ycl#预测列
        self.datapath=pd.DataFrame(datapath)#数据路径
    def fbdata(self):
        data=self.datapath
        data=data[['date',self.ycl]]
        data.columns=['ds','y']
        data['y']=data['y'].apply(lambda x:float(x))
        data['ds']=pd.to_datetime(data['ds'])
        return data
    def fit(self,fbdata):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,changepoints=None)
        m.add_country_holidays(country_name="CN")
        mfit=m.fit(fbpredict.fbdata(self))
        return m
    def predict(self,fit,fbdata):
        future=fbpredict.fit(self,fbpredict.fbdata(self)).make_future_dataframe(periods=int(self.day))
        forecast = fbpredict.fit(self,fbpredict.fbdata(self)).predict(future)
        forecast_jg=pd.merge(forecast,fbpredict.fbdata(self),on='ds',how='left')#模型评估#模型保存
        forecast_jg=forecast_jg.set_index('ds')
        forecast_jg=forecast_jg.astype('float')
        forecast_jg['e']=forecast_jg['y']-forecast_jg['yhat']
        forecast_jg_=forecast_jg[['y','yhat']]
        forecast_jg_['yhat']=forecast_jg_['yhat'].apply(lambda x:round(x,2))
        train_len=len(fbpredict.fbdata(self)['y'])
        rmse = np.sqrt(np.mean(np.power(fbpredict.fbdata(self)["y"] - forecast["yhat"].head(train_len),2)))
        mae=mean_absolute_error(fbpredict.fbdata(self)["y"],forecast["yhat"]).head(train_len)
        mse=mean_squared_error(fbpredict.fbdata(self)["y"],forecast["yhat"]).head(train_len)
        r2=r2_score(fbpredict.fbdata(self)["y"],forecast["yhat"]).head(train_len)
        model_estimate=pd.DataFrame({'prophet':[rmse,mae,mse,r2]},index=['rmse','mae','mse','r2'])
        return forecast_jg_,model_estimate
d=fbpredict(day,ycl,data)#初始化
ddata=d.fbdata()
d0=d.predict(d.fit(ddata),ddata)
result=d0
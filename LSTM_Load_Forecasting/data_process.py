# -*- coding: UTF-8 -*-
import os, random, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from args import us_args_parser,ms_args_parser,mm_args_parser
from sklearn.preprocessing import MinMaxScaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_datas(df):
    """
    :return:
    """
    #path = os.path.dirname(os.path.realpath(__file__)) + '/data/shengzhu.xlsx'
    df = pd.read_excel(df)
    df = df.set_index('date')
    return df


class MyDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq_mm(args,seq_len, B, num,train_split,test_split):
    print('data processing...')
    args=mm_args_parser()
    dataset = load_datas(args.df).iloc[:,int(args.yucestart):int(args.yuceend)+1]
    #数据切分
    train_data = dataset[:int(len(dataset) * args.train_split)]
    val_data = dataset[int(len(dataset) * args.train_split):int(len(dataset) * args.test_split)]
    test_data = dataset[int(len(dataset) * args.test_split):len(dataset)]#shape 为(391, 3)
    global scaler
    #归一化，shape不变,但类型为numpy.ndarray
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train_data)
    val = scaler.transform(val_data)
    test = scaler.transform(test_data)#shape 为(391, 3)
    def process(data, batch_size,step_size):
        #load = data.iloc[:,int(args.yucestart):int(args.yuceend)+1]
        #load1= scaler.fit_transform(load)#load.values的shape是(1172，int(args.yuceend)-int(args.yucestart))
        seq = []
        for i in range(0,len(data) - seq_len-num,num):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x=data[j,0:int(args.input_size)]#一个冒号的shape为(int(args.yuceend)+1-int(args.yucestart),)
                train_seq.append(x)#仍是列表
            #print(np.array(train_seq).shape)#(args.seq_len,int(args.yuceend)+1-int(args.yucestart))
            for b in range(i + seq_len, i + seq_len + num):
                train_label.append(data[b,int(args.yuceend)-int(args.yucestart)])
            #print(np.array(train_label).shape)#(num,)
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)#shape为([4])#确保为一维
            seq.append((train_seq, train_label))
        #print(len(seq)) 例如这里test的shape为(391,3)所以seq的长度为(391-seq_len)//num,即为90

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        return seq

    Dtr = process(train, B, step_size=num)
    Val = process(val, B, step_size=num)
    Dte = process(test, B, step_size=num)
    return (
     Dtr, Val, Dte, scaler)


def nn_seq_ms(args,seq_len, B,train_split,test_split):
    '''
    data.iloc[:,int(args.yucestart):int(args.yuceend)+1] 这里的维度就会为2维,shape为(len(data),int(args.yuceend)+1-int(args.yucestart))
    minmaxscaler的使用方式：scaler.fit_transform(data.iloc[])
    :param seq_len: 预测步长
    :param B: batch size
    :param train_split: 训练集切分比例
    :param test_split: 测试集切分比例
    :return:train_seq 的shape为(30,1)(seq_len,input_size) train_label的shape为(1)(output_size)
    '''
    print('data processing...')
    args=ms_args_parser()
    dataset = load_datas(args.df).iloc[:,int(args.yucestart):int(args.yuceend)+1]
    #数据切分
    train_data = dataset[:int(len(dataset) * args.train_split)]
    val_data = dataset[int(len(dataset) * args.train_split):int(len(dataset) * args.test_split)]
    test_data = dataset[int(len(dataset) * args.test_split):len(dataset)]#shape 为(391, 3)
    global scaler
    #归一化，shape不变,但类型为numpy.ndarray
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train_data)
    val = scaler.transform(val_data)
    test = scaler.transform(test_data)#shape 为(391, 3)

    def process(data, batch_size):
        #load = data.iloc[:,int(args.yucestart):int(args.yuceend)+1]
        #load1= scaler.fit_transform(load)#load.values的shape是(1172，int(args.yuceend)-int(args.yucestart))
        seq = []
        for i in range(len(data) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x=data[j,0:int(args.input_size)]#一个冒号的shape为(int(args.yuceend)+1-int(args.yucestart),)
                train_seq.append(x)#仍是列表
            #print(np.array(train_seq).shape)#(args.seq_len,int(args.yuceend)+1-int(args.yucestart))

            train_label.append(data[(i + seq_len),int(args.yuceend)-int(args.yucestart)])
            train_seq = torch.FloatTensor(train_seq)#可将list转化成tensor，np.array(train_seq)后的shape就为torch.Size([seq_len,int(args.yuceend)+1-int(args.yucestart)])
            train_label = torch.FloatTensor(train_label).view(-1)#shape为torch.Size([1])
            seq.append((train_seq, train_label))
        print(len(seq))
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        return seq

    Dtr = process(train, B)
    Val = process(val, B)
    Dte = process(test, B)
    return (
     Dtr, Val, Dte,scaler)


def nn_seq_us(args,seq_len, B,train_split,test_split):
    '''
    这里[:int(len(dataset) * train_split)][data.columns[args.yucel]的values因为是单变量，所以shape为（len(dataset) * train_split,)，如需fit_transform需要 pd.DataFrame()扩展维度
    label这里因为view(-1)是确保为1维，例如如果使用minmaxscaler后增加的维度在label这里需要删除
    minmaxscaler的单变量使用方式：scaler.fit_transform(pd.DataFrame(data.iloc[].values))
    :param seq_len: 预测步长
    :param B: batch size
    :param train_split: 训练集切分比例
    :param test_split: 测试集切分比例
    :return:train_seq 的shape为(30,1)(seq_len,input_size) train_label的shape为(1)(output_size)
    '''
    print('data processing...')
    args=us_args_parser()
    dataset = load_datas(args.df)
    train = dataset[:int(len(dataset) * train_split)]
    val = dataset[int(len(dataset) * train_split):int(len(dataset) * test_split)]
    test = dataset[int(len(dataset) * test_split):len(dataset)]
    m, n = np.max(train[train.columns[args.yucel]]), np.min(train[train.columns[args.yucel]])#第一列的最大最小值
    #dataset.iloc[:, args.yucestart:args.yuceend]

    def process(data,batch_size):
        #load = data[data.columns[args.yucel]]#预测列
        load = data.iloc[:, int(args.yucel)]
        #方法一：如果选择平均化方式，因为这里是单变量，values的shape为（长度，）所以在创建数据时需要扩展维度x=[load[j]]为2维
        load0 = (load- n) / (m - n) #load.shape为（1178，）
        load0 = load0.tolist()  # 对于单变量预测来说，是否tolist取数并无关系，可注释掉
        #方法二：如果选择minmaxscaler方式，注意这里必须要pd.DataFrame()增加维度，而load1的shpae为（长度，1），所以在创建数据时不需要扩展维度直接为x=load[j]
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #load1= scaler.fit_transform(pd.DataFrame(load.values))
        #load1 = load1.tolist()#第一列数据归一化，是否tolist取数并无关系，可注释掉
        seq = []
        for i in range(len(load) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                #x=load1[j]
                x=[load0[j]]
                train_seq.append(x)
            #print(np.array(train_seq).shape)#shape为（30，1）
            train_label.append(load0[(i + seq_len)])#前24个数预测第25个
            train_seq = torch.FloatTensor(train_seq)#torch.FloatTensor(train_seq)类型转换, 将list ,numpy转化为tensor;等价于torch.from_numpy(np.array(train_seq))
            train_label = torch.FloatTensor(train_label).view(-1)#确保为一维
            seq.append((train_seq, train_label))#seq[0]代表第一个预测的x和y
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)#torch格式的数据加载
        return seq


    Dtr = process(train, B)
    Val = process(val, B)
    Dte = process(test, B)
    return (
     Dtr, Val, Dte,m,n)

def data_gy(data):
    m, n = np.max(data), np.min(data)
    data=(data-n)/(m-n)
    return data
def insertscaler(data,args):
    '''
    :param data: np.array后的，np。reshape可改变列表维度
    :return: 逆变化后的预测列，因为我的预测列是最后一列，所以这里是-1
    '''
    data_array=np.repeat(data,int(args.yuceend)+1-int(args.yucestart),axis=-1)#必须要改变形状才能inverse_transform
    data_ = scaler.inverse_transform(np.reshape(data_array,(len(data),int(args.yuceend)+1-int(args.yucestart))))[:,-1]#逆变化的第一列
    return data_
def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))
# okay decompiling data_process.cpython-37.pyc

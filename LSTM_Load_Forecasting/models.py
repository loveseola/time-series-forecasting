
# -*- coding: UTF-8 -*-
from torch import nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size#单变量还是多变量
        self.hidden_size = hidden_size#隐藏层节点个数
        self.num_layers = num_layers#层数
        self.output_size = output_size
        self.num_directions = 1#单向
        self.batch_size = batch_size
        self.lstm = nn.LSTM((self.input_size), (self.hidden_size), (self.num_layers), batch_first=True)#将batch_size提前
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]#输入到lstm中的input的shape应该 input(batch_size,seq_len,input_size)
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))#(隐状态h_n，单元状态c_n)
        pred = self.fc(output)
        pred = pred[:, -1, :]
        return pred


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM((self.input_size), (self.hidden_size), (self.num_layers), batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)#全连接层

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]#用pytorch输入进去本身是(seq_len,batch_size,input_size)
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        output = output.contiguous().view(batch_size, seq_len, self.num_directions, self.hidden_size)
        output = torch.mean(output, dim=2)
        pred = self.linear(output)
        pred = pred[:, -1, :]#pred的shape为(batch_size, seq_len, output_size)。假设我们用前24个小时(1~24)预测后2个小时的负荷(25~26)，那么seq_len=24, output_size=2。根据LSTM的原理，最终的输出中包含了所有位置的预测值，也就是((2 3), (3 4), (4 5)...(25 26))。很显然我们只需要最后一个预测值，即pred[:, -1, :]。
        return pred
# okay decompiling models.cpython-37.pyc

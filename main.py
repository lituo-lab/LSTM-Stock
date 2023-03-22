import matplotlib.pyplot as plt
import numpy as np
import tushare as ts
import torch
from torch import nn
import datetime
import time


def create_dataset(data, days_for_train=5) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集
        
        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。

        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    dataset_x, dataset_y= [], []
    for i in range(len(data)-days_for_train):
        _x = data[i:(i+days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i+days_for_train])
    return (np.array(dataset_x), np.array(dataset_y))


class LSTM_Regression(nn.Module):  
    #(input_size= 5, hidden_size=8, output_size=1, num_layers=2).
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x



if __name__ == '__main__':
    t0 = time.time()
    data_close = ts.get_k_data('000001', start='2019-01-01', index=True)['close'].values  # 取上证指数的收盘价的np.ndarray 而不是pd.Series
    data_close = data_close.astype('float32')  # 转换数据类型
    plt.figure()
    plt.plot(data_close)
    # plt.savefig('data.png', format='png', dpi=200)
    # plt.close()

    DAYS_FOR_TRAIN=10
    # 将价格标准化到0~1
    max_value = np.max(data_close)
    min_value = np.min(data_close)
    data_close = (data_close - min_value) / (max_value - min_value)

    dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)

    # 划分训练集和测试集，70%作为训练集
    train_size = int(len(dataset_x) * 0.7)

    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]

    # 将数据改变形状，RNN 读入的数据维度是 (seq_size, batch_size, feature_size)
    train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
    train_y = train_y.reshape(-1, 1, 1)
    
    # 转为pytorch的tensor对象
    train_x = torch.from_numpy(train_x).to('cuda')
    train_y = torch.from_numpy(train_y).to('cuda')

    model = LSTM_Regression(input_size= DAYS_FOR_TRAIN, hidden_size=8, output_size=1, num_layers=2).to('cuda')

    
    loss_function = nn.MSELoss().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for i in range(1000):                   
        out = model(train_x)
        loss = loss_function(out, train_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with open('log.txt', 'a+') as f:
            f.write('{} - {}\n'.format(i+1, loss.item()))
        if (i+1) % 1 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.item()))


model2 = model.to('cpu').eval() # 转换成测试模式
# model.load_state_dict(torch.load('model_params.pkl'))  # 读取参数

# 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN 填充使长度相等再作图
dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
dataset_x = torch.from_numpy(dataset_x)

pred_test = model2(dataset_x) # 全量训练集的模型输出 (seq_size, batch_size, output_size)
pred_test = pred_test.view(-1).data.numpy()
pred_test = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_test))  # 填充0 使长度相同
assert len(pred_test) == len(data_close)

#%%
plt.figure()
plt.plot(pred_test, 'r', label='prediction')
plt.plot(data_close, 'b', label='real')
plt.plot((train_size, train_size), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
# plt.xlim((800,900))
# plt.ylim((0.25,1))
plt.legend(loc='best')







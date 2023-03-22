# 使用LSTM预测股票



## 问题概述



使用Tushare获得股票价格信息作为数据集，Tushare是一个免费、开源的python财经数据接口包。

学习RNN、LSTM以及GRN网络结构和函数接口使用，学习循环神经网络的使用。



## 模型介绍

网络使用LSTM+MLP结构，(input_size= 5, hidden_size=8, output_size=1, num_layers=2)，

输入(seq_size, batch_size, feature_size=5)，feature为前5天的价格信息；

输出(seq_size, batch_size, output_size=1)，output为当天的价格信息。



grn&lstm文件夹为附带的学习使用grn和lstm的可运行实例。



## 参考链接

https://blog.csdn.net/m0_37758063/article/details/117995469
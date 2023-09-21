# torch 实现 FNN 解决 二分类问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

import torch
import numpy as np
import random
import time
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim  # 优化器
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

"""
疑问、
1、隐藏层为啥是：self.num_hiddens[-1]，取一个值呢，不应该是列表么。
2、没有初始化权重和偏执。答：torch 里面有默认的。
"""

# 训练集
train_dataset = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST/train", train=True,
                                                  transform=transforms.ToTensor(),
                                                  download=False)

# 测试集
test_dataset = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST/test", train=False,
                                                 transform=transforms.ToTensor(),
                                                 download=False)

# 把 dataset 放入 DataLoader
batch_size = 256
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 2&3)定义模型及其前向传播过程:一层神经网络 一层隐藏层
# 2) 定义模型
class MyTenNet(nn.Module):
    def __init__(self, num_hiddenlayer=1, num_inputs=28 * 28, num_hiddens=[256], num_outs=10, act='relu'):
        super(MyTenNet, self).__init__()
        # 设置输入层（特征数）、隐藏层（神经元个数）、输出层的节点数202320
        self.num_inputs, self.num_hiddens, self.num_outputs = num_inputs, num_hiddens, num_outs  # 输入层、隐藏层、输出层 的 神经元个数

        # 定义模型结构 *****: 定义激活函数，输入层变成一个一维向量、隐藏层激活函数 relu 函数、输出层激活函数
        # 模型输入，拉平成向量
        self.input_layer = nn.Flatten()
        # 若只有一层隐藏层
        if num_hiddenlayer == 1:
            self.hidden_layers = nn.Linear(self.num_inputs, self.num_hiddens[-1])  # 隐藏层
        else:
            self.hidden_layers = nn.Sequential()
            self.hidden_layers.add_module("hidden_layer1", nn.Linear(self.num_inputs, self.num_hiddens[0]))
            for i in range(0, num_hiddenlayer - 1):
                name = str('hidden_layer' + str(i + 2))
                self.hidden_layers.add_module(name, nn.Linear(self.num_hiddens[i], self.num_hiddens[i + 1]))
        self.output_layer = nn.Linear(self.num_hiddens[-1], self.num_outputs)  # 模型输出:

        # 指代需要使用什么样子的激活函数
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'elu':
            self.act = nn.ELU()
        print(f'你本次使用的激活函数为 {act}')

    # 3) 定义模型前向传播过程
    # 多分类，为啥没有看到使用 softmax 非线性决策函数呢。
    def forward(self, x):
        flatten_input = self.input_layer(x)
        hidden_output = self.act(self.hidden_layers(flatten_input))
        final_output = self.output_layer(hidden_output)
        return final_output


model = MyTenNet()  # 模型
print('model:', model)


# 将训练过程定义为一个函数，方便实验三和实验四调用
def train_and_test_fun(model=model):
    myModel = model
    print(myModel)
    loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=1e-3)  # 梯度下降法
    epochs = 40
    train_all_loss = []  # 记录训练集上得loss变化
    test_all_loss = []  # 记录测试集上的loss变化
    train_acc, test_acc = [], []
    begin_time = time.time()
    for epoch in range(epochs):
        train_l, train_epoch_count, test_epoch_count = 0, 0, 0
        for data, label in train_loader:
            pre = myModel(data)
            l = loss(pre, label.view(-1))  # 计算每次的损失值
            optimizer.zero_grad()
            l.backward()  # 反向传播
            optimizer.step()

            train_l += l.item()
            train_epoch_count += (pre.argmax(dim=1) == label).sum()
            # print(train_each_loss)
        train_all_loss.append(train_l)  # 添加损失值到列表中
        train_acc.append(train_epoch_count.cpu() / len(train_dataset))
        with torch.no_grad():
            test_loss, test_epoch_count = 0, 0
            for X, y in test_loader:
                p = myModel(X)
                ll = loss(p, y.view(-1))
                test_loss += ll.item()
                test_epoch_count += (p.argmax(dim=1) == y).sum()
            test_all_loss.append(test_loss)
            test_acc.append(test_epoch_count.cpu() / len(test_dataset))
        if epoch == 0 or (epoch + 1) % 4 == 0:
            print('epoch: %d | train loss:%.5f | test loss:%.5f | train acc:%5f test acc:%.5f:' % (
                epoch + 1, train_all_loss[-1], test_all_loss[-1],
                train_acc[-1], test_acc[-1]))

    end_time = time.time()
    print("torch.nn实现前馈网络-多分类任务 %d轮 总用时: %.3fs" % (epochs, end_time - begin_time))
    # 返回训练集和测试集上的 损失值 与 准确率
    return train_all_loss, test_all_loss, train_acc, test_acc


def ComPlot(datalist, title='1', ylabel='Loss', flag='act'):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(datalist[0], label='Tanh' if flag == 'act' else '[128]')
    plt.plot(datalist[1], label='Sigmoid' if flag == 'act' else '[512 256]')
    plt.plot(datalist[2], label='ELu' if flag == 'act' else '[512 256 128 64]')
    plt.plot(datalist[3], label='Relu' if flag == 'act' else '[256]')
    plt.legend()
    plt.grid(True)

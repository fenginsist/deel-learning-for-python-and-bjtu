# 利用torch.nn实现 softmax 回归在Fashion-MNIST数据集上进行训练和测试，
# 并从loss，训练集以及测试集上的准确率等多个角度对结果进行分析
# https://blog.csdn.net/qq_37534947/article/details/108179408


# 注意的点
# 1、loss记得 .sum()
# 2、先梯度清零

# 问题：
# 1、定义模型时：为啥这里没有像 作业3一样，先进行 线性判别 再进行 非线性决策呢？只看到了线性函数，没有看到决策函数。困惑

import torch
import torch.nn as nn
from torch.nn import init as Init  # 调用𝐢𝐧𝐢𝐭. 𝐧𝐨𝐫𝐦𝐚𝐥模块通过正态分布对线性 回归中的权重和偏差进行初始化
import numpy as np
import torchvision
import torchvision.transforms as transforms

# 下载的数据
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False,
                                                transform=transforms.ToTensor())

# 读取数据
batch_size = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)

# 构建模型
input_num = 784
output_num = 10


# --------------------------------------注：这里和之前的logistic实现几乎一样，主要是对于该模型的输出从1变成10.
class softmax_net(nn.Module):
    def __init__(self, input_num, output_num):
        super(softmax_net, self).__init__()
        self.linear = nn.Linear(input_num, output_num)

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y


net = softmax_net(input_num, output_num)

# 初始化权重和偏差
Init.normal_(net.linear.weight, mean=0, std=0.01)
Init.constant_(net.linear.bias, val=0)

# 损失函数
# nn模块实现交叉熵损失函数--包含了softmax函数
loss = nn.CrossEntropyLoss()

# 定义优化函数-小批量梯度下降
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 迭代
epochs = 10

for epoch in range(epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        pre_y = net(X)
        l = loss(pre_y, y).sum()
        optimizer.zero_grad()  # 先梯度清零
        l.backward()
        optimizer.step()
        # 计算每个epoch的loss
        train_l_sum += l.item()
        n += y.shape[0]
    print('epoch %d, loss %.4f, train acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n))

# 实验2中 应对过拟合问题的常用方法—— L2范数正则化
# 手动实现 L2范数正则化

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import sys

sys.path.append("..")
from matplotlib import pyplot as plt
from IPython import display

import matplotlib_inline

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

# 1 生成数据集
features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels = torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float, requires_grad=True)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]
print(train_features[0][:5])  # 输出第一个样本特征向量的 前五维的元素
print(train_labels[0])


# 1 定义模型
def linear(X, w, b):
    return torch.mm(X, w) + b


# 2 损失函数：定义均方误差
def squared_loss(y_hat, y):
    # 返回的是向量，注意: pytorch里的MSELoss 并没有除以2
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2


# 3 损失函数后使用： 手动实现𝑳𝟐范数正则化
# 3.1 定义随机初始化模型参数的函数
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)  # num_inputs=200
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 3.2 定义 𝐿2范数惩罚项，返回的是一个float的数
def penalty_12(w):
    return (w ** 2).sum() / 2


# 4 优化函数 定义随机梯度下降函数
def SGD(params, lr):
    for param in params:
        # 注意这里参数赋值用的是 param.data
        param.data -= lr * param.grad


def Draw_Loss_Curve(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    # display.set_matplotlib_formats('svg')
    # 上面的报错，方法被舍弃了，网上说用下面这个包和方法
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)


batch_size, num_opochs, lr = 1, 100, 0.003
net, loss = linear, squared_loss

# 划分数据集
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


# 训练模型
def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    train_l_sum = 0
    for epoch in range(num_opochs):
        for X, y in train_iter:
            pre_y = net(X, w, b)
            # 添加了 L2 范数惩罚项
            l = (loss(pre_y, y) + lambd * penalty_12(w)).sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            SGD([w, b], lr)

            train_l_sum += l.item()
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
        print('epoch %d, train loss %.4f' % (epoch + 1, train_l_sum))
    Draw_Loss_Curve(range(1, num_opochs + 1), train_ls, 'epochs', 'loss', range(1, num_opochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())


# 𝜆 = 0(即不使用𝐿"范数正则化)时的实验 结果，出现了过拟合的现象。
# fit_and_plot(0)
# 𝜆 = 3(即使用𝐿"范数正则化)时的实验结果 ，一定程度的缓解了过拟合。同时可以看到 参数𝐿"范数变小，参数更接近0。
fit_and_plot(3)

# 手动实现 线性回归实现


import matplotlib_inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

num_inputs = 2  # 特征数
num_examples = 1000  # 样本数
true_w = [2, -3.4]  # 真实的权重w
true_b = 4.2  # 真实的偏差c
# 需要得出真实的标签 label
# np.random.normal(0, 1, (num_examples, num_inputs)) ： 生成了 1000行 * 2列的 0-1之间的 矩阵数据（非tensor 为np数学数组）。
# 测试：x = np.random.normal(0, 1, (5, 2))。np.random.normal() 返回一组符合高斯分布的概率密度随机数。
# 特征 feature的数据，会生成一个 1000 行 * 2列的矩阵数据代表特征feature
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
# 标签 label计算，1*2 的矩阵和 2*1的矩阵相乘 + 加上偏差。features[:, 0] 取的是第一列，features[:, 1] 取第二列
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)  # 加上 噪声


# 图没出来，不知道为啥
def use_svg_display():
    # 用矢量图显示
    # display.set_matplotlib_formats('svg')
    matplotlib_inline.backend_inline.set_matplotlib_formats()


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


# 初始化权重和偏织
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)  # 默认初始化为0


# 构建模型
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 损失函数：平方损失函数（预测值减去真实值）
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 优化算法：sgd 函数实现了小批量随机梯度下降算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


lr = 0.03
epochs = 3
batch_size = 10
net = linreg
loss = squared_loss

for i in range(epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次
    for X, y in data_iter(batch_size, features, labels):  # x和y分别是小批量样本的特征和标签
        pred_y = net(X, w, b)  # 通过模型，算出预测 标签y 值
        l = loss(pred_y, y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
        w.grad.data.zero_()  # 梯度清零
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (i + 1, train_l.mean().item()))

print(true_w, '---', w)
print(true_b, '---', b)

print("end")

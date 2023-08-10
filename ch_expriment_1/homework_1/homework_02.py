# 从0实现logistic回归(只借助Tensor和Numpy相关的库)
# https://blog.csdn.net/weixin_41939870/article/details/120611800
# https://pytorch.org/docs/1.2.0/torch.html#torch.log
# 二分类问题 一般采用 logistics 函数，采用交叉熵损失函数，梯度下降 做参数的优化

# 注意的点
# 1、loss记得 .sum()
# 2、需要注意的是矩阵相乘利用的是torch.mm，而矩阵对应位置相乘是利用的.mul，另外利用 view()函数统一为同型张量。

# 问题
# 1、debug 模型 时，X 是tensor，里面一层层的 H 和 T ，这是什么东西呢？困惑

import torch
import random
import numpy as np

# 训练集
n_data = torch.ones(100, 2)  # 数据的基本形态
x1 = torch.normal(2 * n_data, 1)  # shape=(50, 2)
y1 = torch.zeros(100)  # 类型0 shape=(50)
x2 = torch.normal(-2 * n_data, 1)  # shape=(50, 2)
y2 = torch.ones(100)  # 类型1 shape=(50)

# 注意 x, y 数据的数据形式一定要像下面一样 (torch.cat 是合并数据)
train_features = torch.cat((x1, x2), 0).type(torch.FloatTensor)  # torch.Size([100, 2])
train_labels = torch.cat((y1, y2), 0).type(torch.FloatTensor)  # torch.Size([100, 2])

# 测试集
n_data = torch.ones(50, 2)  # 数据的基本形态
x1 = torch.normal(2 * n_data, 1)  # shape=(50, 2)
y1 = torch.zeros(50)  # 类型0 shape=(50)
x2 = torch.normal(-2 * n_data, 1)  # shape=(50, 2)
y2 = torch.ones(50)  # 类型1 shape=(50)

# 注意 x, y 数据的数据形式一定要像下面一样 (torch.cat 是合并数据)
test_features = torch.cat((x1, x2), 0).type(torch.FloatTensor)  # torch.Size([100, 2])
test_labels = torch.cat((y1, y2), 0).type(torch.FloatTensor)  # torch.Size([100, 2])


# 数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


# 构建 Logistic 模型
def logistic_regression(X, w, b):
    # torch.mm(X, w) + b ：这个就是 线性判别函数，外面的logistic函数就是 非线性决策函数 。通过 决策函数实现 二分类
    return 1 / (1 + torch.exp(-1 * torch.mm(X, w) + b))


# 初始化 权重 和 偏差
num_inputs = 2
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)
print('w:', w)
print('w.size():', w.size())  # torch.Size([2, 1]) 二两一列
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
print('b:', b)
print('b.size():', b.size())  # torch.Size([1])


# 损失函数：手动实现二元交叉熵损失函数
def logistic_loss(y_hat, y):
    # 这个是 BCE_loss 函数
    # return -1 * (y * torch.log10(y_hat) + (1 - y) * torch.log10(1 - y_hat))
    y = y.view(y_hat.size())  # 变成了行向量
    return -y.mul(torch.log(y_hat)) - (1 - y).mul(torch.log(1 - y_hat))


# 注：Logistic回归的模型定义,以及二元交叉熵的损失函数和优化函数。
# 需要注意的是矩阵相乘利用的是torch.mm，而矩阵对应位置相乘是利用的.mul，另外利用 view()函数统一为同型张量。

# 优化器
# 手动定义sgd优化函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


# 测试集准确率
def evaluate_accuracy():
    acc_sum, n = 0.0, 0
    for X, y in data_iter(batch_size, test_features, test_labels):
        # print(len(X)) 小批量数据集 每个X中有 256个图像
        # print((net(X).argmax(dim=1)==y).float().sum().item())
        y_hat = net(X, w, b)
        y_hat = torch.squeeze(torch.where(y_hat > 0.5, torch.tensor(1.0), torch.tensor(0.0)))
        acc_sum += (y_hat == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 迭代
net = logistic_regression
loss = logistic_loss
lr = 1e-4
epoch = 4
batch_size = 10

print("开始训练啦～～～")
for i in range(epoch):
    train_l_num, train_acc_num, n = 0.0, 0.0, 0
    for X, y in data_iter(batch_size, train_features, train_labels):
        pre_y = net(X, w, b)  # pre_y：预测的标签y值，因为 batch_size=10，所以 为 10行1列的tensor
        l = loss(pre_y, y).sum()
        # 计算梯度 反向传播
        l.backward()
        sgd((w, b), lr, batch_size)
        # reset parameter gradient梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
        # 计算每个epoch的loss
        train_l_num += l.item()

        # 计算训练样本的准确率
        pre_y = torch.squeeze(torch.where(pre_y > 0.5, torch.tensor(1.0), torch.tensor(0.0)))
        train_acc_num += (pre_y == y).sum().item()

        # 每一个epoch的所有样本数
        n += y.shape[0]
    test_acc = evaluate_accuracy()
    print('epoch %d, loss %.4f,train_acc %f,test_acc %f' % (i + 1, train_l_num / n, train_acc_num / n, test_acc))

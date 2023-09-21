# 手动实现 FNN 解决 回归问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

import torch
import numpy as np
import random
import time

true_w, true_b = torch.ones(500, 1) * 0.0056, 0.028  # 真实的权重w、真实的偏差c

# 训练集
train_feature_line = torch.tensor(np.random.normal(0, 1, (7000, 500)), dtype=torch.float)
train_labels_line = torch.matmul(train_feature_line, true_w) + true_b  # matmul：矩阵维度相同，对应相乘。
train_labels_line += torch.tensor(np.random.normal(0, 0.01, size=train_labels_line.size()), dtype=torch.float,
                                  requires_grad=True)

# 测试集
test_feature_line = torch.tensor(np.random.normal(0, 1, (3000, 500)), dtype=torch.float)
test_labels_line = torch.matmul(test_feature_line, true_w) + true_b  # matmul：矩阵维度相同，对应相乘。
test_labels_line += torch.tensor(np.random.normal(0, 0.01, size=test_labels_line.size()), dtype=torch.float,
                                 requires_grad=True)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


# 2&3)定义模型及其前向传播过程:一层神经网络 一层隐藏层
# 2) 定义模型
class MyNet():
    def __init__(self):
        # 设置输入层（特征数）、隐藏层（神经元个数）、输出层的节点数
        num_inputs, num_hiddens, num_outputs = 500, 256, 1  # 输入层、隐藏层、输出层 的 神经元个数
        # 以下是一层神经网络（隐藏层）的隐藏层和输出层的权重和偏差。
        w1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float, requires_grad=True)
        b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
        w2 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float, requires_grad=True)
        b2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
        self.params = [w1, b1, w2, b2]

        # 定义模型结构 *****: 定义激活函数，输入层变成一个一维向量、隐藏层激活函数 relu 函数、输出层激活函数
        self.input_layer = lambda x: x.view(x.shape[0], -1)  # 模型输入，拉平成向量
        self.hidden_layer = lambda x: self.my_ReLu(torch.matmul(x, w1.t()) + b1)  # 隐藏层，使用 relu 函数。
        self.output_layer = lambda x: torch.matmul(x, w2.t()) + b2  # 模型输出:

    def my_ReLu(self, x):
        return torch.max(input=x, other=torch.tensor(0.0))

    # 3) 定义模型前向传播过程
    # 多分类，为啥没有看到使用 softmax 非线性决策函数呢。
    def forward(self, x):
        flatten_input = self.input_layer(x)
        hidden_output = self.hidden_layer(flatten_input)
        final_output = self.output_layer(hidden_output)
        return final_output


# 损失函数
def MSE_loss(pred, true):
    ans = torch.sum((true - pred) ** 2) / len(pred)
    print('ans:', ans)
    return ans


# 均方误差损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 优化算法
def mySGD(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


model = MyNet()
loss = squared_loss
sgd = mySGD
lr = 0.05
batch_size = 128
epochs = 40
train_all_loss = []  # 记录训练集上得loss变化
test_all_loss = []  # 记录测试集上的loss变化
begin_time = time.time()
for epoch in range(epochs):
    train_l = 0
    for data, label in data_iter(batch_size, train_feature_line, train_labels_line):
        pre = model.forward(data)
        l = loss(pre.view(-1, 1), label.view(-1, 1)).sum()  # 计算每次的损失值
        l.backward()  # 反向传播
        sgd(model.params, lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
        # 梯度清零
        for param in model.params:
            param.grad.data.zero_()

        train_l += l.item()
        # print(train_each_loss)
    train_all_loss.append(train_l)  # 添加损失值到列表中
    with torch.no_grad():
        test_loss = 0
        for X, y in data_iter(batch_size, test_feature_line, test_labels_line):
            p = model.forward(X)
            ll = loss(p, y).sum()
            test_loss += ll.item()
        test_all_loss.append(test_loss)
    if epoch == 0 or (epoch + 1) % 4 == 0:
        print('epoch: %d | train loss:%.5f | test loss:%.5f' % (epoch + 1, train_all_loss[-1], test_all_loss[-1]))
end_time = time.time()
print("手动实现前馈网络-回归实验 %d轮 总用时: %.3fs" % (epochs, begin_time - end_time))

# 利用 torch.nn 实现 logistic 回归在人工构造的数据集上进行训练和测试，并对结果进行分析， 并从loss以及训练集上的准确率等多个角度对结果进行分析

# 问题1：通过 torch 实现logistic 时，权重和偏差的初始化，时初始化 Liner 内部的 w b 吗？
# 问题2：

import torch
import torch.nn as nn
from torch.nn import init  # 调用𝐢𝐧𝐢𝐭. 𝐧𝐨𝐫𝐦𝐚𝐥模块通过正态分布对线性 回归中的权重和偏差进行初始化
import torch.utils.data as Data

n_data = torch.ones(50, 2)  # 数据的基本形态
x1 = torch.normal(2 * n_data, 1)  # shape=(50, 2)
y1 = torch.zeros(50)  # 类型0 shape=(50)
x2 = torch.normal(-2 * n_data, 1)  # shape=(50, 2)
y2 = torch.ones(50)  # 类型1 shape=(50)

# 注意 features, labels 数据的数据形式一定要像下面一样 (torch.cat 是合并数据)
features = torch.cat((x1, x2), 0).type(torch.FloatTensor)
labels = torch.cat((y1, y2), 0).type(torch.FloatTensor)

batch_size = 50
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 把 dataset 放入 DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,  # 是否打乱数据 (训练集一般需要进行打乱)
    num_workers=0,  # 多线程来读数据， 注意在Windows下需要设置为0
)


# 构建模型
class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.liner = nn.Linear(n_features, 1)   # 初始化 线性判别函数，里面默认会有  权重和偏差
        self.sm = nn.Sigmoid()                  # 初始化 非线性 决策函数

    # 先 执行 线性判别函数，再执行 非线性决策函数
    def forward(self, x):
        x = self.liner(x)
        x = self.sm(x)
        return x


num_inputs = 2
logistic_model = LogisticRegression(num_inputs)

# 给 模型中 线性判别函数 进行，初始化 权重和 偏差
init.normal_(logistic_model.liner.weight, mean=0, std=0.01)
init.constant_(logistic_model.liner.bias, val=0)               # 也可以直接修改bias的data： net[0].bias.data.fill_(0)
print(logistic_model.liner.weight)
print(logistic_model.liner.weight.size())   # torch.Size([1, 2])
print(logistic_model.liner.bias)
print(logistic_model.liner.bias.size())     # torch.Size([1])

# 损失函数：二元交叉熵损失函数
loss = torch.nn.BCELoss()

# 优化器
## 传入 模型 中的 参数，也是内嵌的
optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3)

epoch = 10
# 迭代
# 注：不同于自定义的模型等训练过程，这里的 梯度清零以及梯度更新 都是在优化器的函数中进行
for i in range(epoch):
    train_l_num, train_acc_num, n = 0.0, 0.0, 0
    for X, y in data_iter:
        pred_y = logistic_model(X)
        l = loss(pred_y, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于logistic_model.zero_grad()
        l.backward()
        optimizer.step()
        train_l_num += l.item()  # 计算每个epoch的loss

        pred_y = torch.squeeze(torch.where(pred_y > 0.5, torch.tensor(1.0), torch.tensor(0.0)))  # 计算训练样本的准确率
        train_acc_num += (pred_y == y).sum().item()

        n += y.shape[0]  # 每一个epoch的所有样本数
    print('epoch %d, loss %.4f,train_acc %f' % (i + 1, train_l_num / n, train_acc_num / n))

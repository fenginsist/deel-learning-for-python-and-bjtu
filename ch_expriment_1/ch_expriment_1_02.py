# torch.nn 实现 线性回归实现
import torch.utils.data as Data
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init   # 调用𝐢𝐧𝐢𝐭. 𝐧𝐨𝐫𝐦𝐚𝐥模块通过正态分布对线性 回归中的权重和偏差进行初始化
import torch.optim as optim # 优化器


num_inputs = 2              # 特征数
num_examples = 1000         # 样本数
true_w = [2, -3.4]          # 真实的权重w
true_b = 4.2                # 真实的偏差c
# 需要得出真实的标签 label
# np.random.normal(0, 1, (num_examples, num_inputs)) ： 生成了 1000行 * 2列的 0-1之间的 矩阵数据（非tensor 为np数学数组）。测试：x = np.random.normal(0, 1, (5, 2))。np.random.normal() 返回一组符合高斯分布的概率密度随机数。
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)  # 特征 feature的数据，会生成一个 1000 行 * 2列的矩阵数据代表特征feature
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b                       # 标签 label计算，1*2 的矩阵和 2*1的矩阵相乘 + 加上偏差。features[:, 0] 取的是第一列，features[:, 1] 取第二列
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)        # 加上 噪声


lr = 0.03
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)

# 把 dataset 放入 DataLoader
data_iter = Data.DataLoader(
    dataset = dataset, # torch TensorDataset format
    batch_size = batch_size, # mini batch size
    shuffle = True, # 是否打乱数据 (训练集一般需要进行打乱)
    num_workers = 2, # 多线程来读数据，注意在Windows下需要设置为0
)


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs) # num_inputs: 特征数
print(net)  # LinearNet( (linear): Linear(in_features=2, out_features=1, bias=True) )

# 权重和偏差进行初始化
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)  #也可以直接修改 bias 的  data:net[0].bias.data.fill_(0)

# 损失函数
loss = nn.MSELoss()

# 优化器
optimizer = optim.SGD(net.parameters(), lr=0.03)

epochs = 3
for i in range(1, epochs + 1):
    for X, y in data_iter:
        pred_y = net(X)
        l = loss(pred_y, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d,loss:%f' % (i, l.item()))
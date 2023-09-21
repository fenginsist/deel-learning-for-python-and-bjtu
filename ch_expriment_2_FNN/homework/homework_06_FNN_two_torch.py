# torch 实现 FNN 解决 二分类问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

import torch
import time
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim  # 优化器


# 训练集
train_feature_2sort = torch.normal(mean=1, std=0.01, size=(7000, 200), dtype=torch.float)
print('train_feature_2sort.shape: ', train_feature_2sort.shape)
train_labels_2sort = torch.zeros(7000, requires_grad=True)
print('train_labels_2sort.shape: ', train_labels_2sort.shape)

# 测试集
test_feature_2sort = torch.normal(mean=-1, std=0.01, size=(3000, 200), dtype=torch.float)
print('test_feature_2sort.shape: ', test_feature_2sort.shape)
test_labels_2sort = torch.zeros(3000, requires_grad=True)
print('test_labels_2sort.shape: ', test_labels_2sort.shape)

# 将训练数据的特征和标签组合
train_dataset = Data.TensorDataset(train_feature_2sort, train_labels_2sort)
test_dataset = Data.TensorDataset(test_feature_2sort, test_labels_2sort)
batch_size = 256

# 把 dataset 放入 DataLoader
train_data_iter = Data.DataLoader(
    dataset=train_dataset,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,  # 是否打乱数据 (训练集一般需要进行打乱)
    num_workers=0,  # 多线程来读数据，注意在Windows下需要设置为0
)
test_data_iter = Data.DataLoader(
    dataset=test_dataset,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,  # 是否打乱数据 (训练集一般需要进行打乱)
    num_workers=0,  # 多线程来读数据，注意在Windows下需要设置为0
)


# 2&3)定义模型及其前向传播过程:一层神经网络 一层隐藏层
# 2) 定义模型
class MyTwoNet(nn.Module):
    def __init__(self):
        super(MyTwoNet, self).__init__()
        # 设置输入层（特征数）、隐藏层（神经元个数）、输出层的节点数202320
        num_inputs, num_hiddens, num_outputs = 200, 256, 1  # 输入层、隐藏层、输出层 的 神经元个数

        # 定义模型结构 *****: 定义激活函数，输入层变成一个一维向量、隐藏层激活函数 relu 函数、输出层激活函数
        self.input_layer = nn.Flatten()  # 模型输入，拉平成向量
        self.hidden_layer = nn.Linear(num_inputs, num_hiddens)  # 隐藏层
        self.output_layer = nn.Linear(num_hiddens, num_outputs)  # 模型输出:
        self.relu = nn.ReLU()

    def logistics(self, x):
        x = 1.0 / (1.0 + torch.exp(-x))
        return x

    # 3) 定义模型前向传播过程
    # 多分类，为啥没有看到使用 softmax 非线性决策函数呢。
    def forward(self, x):
        flatten_input = self.input_layer(x)
        hidden_output = self.relu(self.hidden_layer(flatten_input))
        final_output = self.logistics(self.output_layer(hidden_output))
        return final_output


model = MyTwoNet()  # 模型
print('model:', model)
loss = nn.BCELoss()  # 二元交叉熵损失函数
sgd = optim.SGD(model.parameters(), lr=1e-3)  # 梯度下降法

lr = 0.05
batch_size = 128
epochs = 40
train_all_loss = []  # 记录训练集上得loss变化
test_all_loss = []  # 记录测试集上的loss变化
begin_time = time.time()
for epoch in range(epochs):
    train_l = 0
    for data, label in train_data_iter:
        pre = model(data)
        l = loss(pre, label.view(-1, 1))  # 计算每次的损失值
        sgd.zero_grad()
        l.backward()  # 反向传播
        sgd.step()

        train_l += l.item()
        # print(train_each_loss)
    train_all_loss.append(train_l)  # 添加损失值到列表中
    with torch.no_grad():
        test_loss = 0
        for X, y in test_data_iter:
            p = model(X)
            ll = loss(p, y.view(-1, 1))
            test_loss += ll.item()
        test_all_loss.append(test_loss)
    if epoch == 0 or (epoch + 1) % 4 == 0:
        print('epoch: %d | train loss:%.5f | test loss:%.5f' % (epoch + 1, train_all_loss[-1], test_all_loss[-1]))
end_time = time.time()
print("手动实现前馈网络-二分类实验 %d轮 总用时: %.3fs" % (epochs, begin_time - end_time))

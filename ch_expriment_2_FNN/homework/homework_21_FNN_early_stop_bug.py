# torch 实现 FNN 解决 二分类问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

import torch
import torch.nn as nn

import time
import random

import torchvision
import torchvision.transforms as transforms

from homework_10_FNN_ten_torch_utils import MyTenNet, train_and_test_torch
from homework_01_dataset import train_dataset, test_dataset, train_loader, test_loader

"""
实现早停机制

Fashion-MNIST数据集:
    该数据集包含60,000个用于训练的图像样本和10,000个用于测试的图像样本。
    图像是固定大小(28x28像素)，其值为0到1。为每个图像都被平展并转换为784（28*28）。
"""

# 得到0-60000的下标
index = list(range(len(train_dataset)))
# 利用shuffle随机打乱下标
random.shuffle(index)
# 按照 训练集和验证集 8：2 的比例分配各自下标
train_index, val_index = index[: 48000], index[48000:]

# 由分配的下标得到对应的训练和验证及的数据以及他们对应的标签
train_dataset, train_labels = train_dataset.data[train_index], train_dataset.targets[train_index]
val_dataset, val_labels = train_dataset.data[val_index], train_dataset.targets[val_index]
print('训练集:', train_dataset.shape, train_labels.shape)
print('验证集:', val_dataset.shape, val_labels.shape)

# 构建相应的dataset以及dataloader
T_dataset = torch.utils.data.TensorDataset(train_dataset, train_labels)
V_dataset = torch.utils.data.TensorDataset(val_dataset, val_labels)
T_dataloader = torch.utils.data.DataLoader(dataset=T_dataset, batch_size=128, shuffle=True)
V_dataloader = torch.utils.data.DataLoader(dataset=V_dataset, batch_size=128, shuffle=True)
print('T_dataset', len(T_dataset), 'T_dataloader batch_size: 128')
print('V_dataset', len(V_dataset), 'V_dataloader batch_size: 128')

model = MyTenNet()


def train_and_test_4(model=model, epochs=30, lr=0.01, weight_decay=0.0):
    print(model)
    # 优化函数, 默认情况下weight_decay为0 通过更改weight_decay的值可以实现L2正则化。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-6)
    criterion = nn.CrossEntropyLoss()  # 损失函数
    train_all_loss = []  # 记录训练集上得loss变化
    val_all_loss = []  # 记录测试集上的loss变化
    train_ACC, val_ACC = [], []
    begintime = time.time()
    flag_stop = 0
    for epoch in range(1000):
        train_l, train_epoch_count, val_epoch_count = 0, 0, 0
        for data, labels in T_dataloader:
            # data, labels = data.to(torch.float32).to(device), labels.to(device)
            pred = model(data)
            train_each_loss = criterion(pred, labels.view(-1))  # 计算每次的损失值
            optimizer.zero_grad()  # 梯度清零
            train_each_loss.backward()  # 反向传播
            optimizer.step()  # 梯度更新
            train_l += train_each_loss.item()
            train_epoch_count += (pred.argmax(dim=1) == labels).sum()
        train_ACC.append(train_epoch_count / len(train_dataset))
        train_all_loss.append(train_l)  # 添加损失值到列表中
        with torch.no_grad():
            val_loss, val_epoch_count = 0, 0
            for data, labels in test_dataset:
                # data, labels = data.to(torch.float32).to(device), labels.to(device)
                pred = model(data)
                val_each_loss = criterion(pred, labels)
                val_loss += val_each_loss.item()
                val_epoch_count += (pred.argmax(dim=1) == labels).sum()
            val_all_loss.append(val_loss)
            val_ACC.append(val_epoch_count / len(test_dataset))
            # 实现早停机制
            # 若连续五次验证集的损失值连续增大，则停止运行，否则继续运行，
            if epoch > 5 and val_all_loss[-1] > val_all_loss[-2]:
                flag_stop += 1
                if flag_stop == 5 or epoch > 35:
                    print('停止运行，防止过拟合')
                    break
            else:
                flag_stop = 0
        if epoch == 0 or (epoch + 1) % 4 == 0:
            print('epoch: %d | train loss:%.5f | val loss:%.5f | train acc:%5f val acc:%.5f:' % (
                epoch + 1, train_all_loss[-1], val_all_loss[-1],
                train_ACC[-1], val_ACC[-1]))
    endtime = time.time()
    print("torch.nn实现前馈网络-多分类任务 %d轮 总用时: %.3fs" % (epochs, endtime - begintime))
    # 返回训练集和测试集上的 损失值 与 准确率
    return train_all_loss, val_all_loss, train_ACC, val_ACC


model = MyTenNet(dropout=0.5)
train_all_loss41, test_all_loss41, train_ACC41, test_ACC41 = train_and_test_4(model=model, epochs=40, lr=0.1)

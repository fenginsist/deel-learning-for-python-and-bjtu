# 手动 实现 FNN 解决 二分类问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

import torch

# 暂时使用这个
from homework_10_FNN_ten_hand_utils import MyTenNet, train_and_test

"""
实现 L2 正则化:
    就是损失函数做一个惩罚。lambd 作为惩罚系数

Fashion-MNIST数据集:
    该数据集包含60,000个用于训练的图像样本和10,000个用于测试的图像样本。
    图像是固定大小(28x28像素)，其值为0到1。为每个图像都被平展并转换为784（28*28）。
"""


model = MyTenNet(dropout=0)  # 模型
print('model:', model)
train_all_loss, test_all_loss, train_ACC, test_ACC = train_and_test(model=model, epochs=20, lr=0.01, L2=True, lambd=0)
print('-----', train_all_loss, test_all_loss, train_ACC, test_ACC)


model_2 = MyTenNet(dropout=0)  # 模型
print('model:', model)
train_all_loss_2, test_all_loss_2, train_ACC_2, test_ACC_2 = train_and_test(model=model_2, epochs=20, lr=0.01, L2=True, lambd=1)
print('-----', train_all_loss_2, test_all_loss_2, train_ACC_2, test_ACC_2)
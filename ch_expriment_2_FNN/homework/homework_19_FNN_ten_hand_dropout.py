# 手动 实现 FNN 解决 二分类问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

import torch

# 暂时使用这个
from homework_10_FNN_ten_hand_utils import MyTenNet, train_and_test
from homework_01_dataset import picture

"""
手动实现 dropout

Fashion-MNIST数据集:
    该数据集包含60,000个用于训练的图像样本和10,000个用于测试的图像样本。
    图像是固定大小(28x28像素)，其值为0到1。为每个图像都被平展并转换为784（28*28）。
"""


model = MyTenNet(dropout=0.0)  # 模型 当 dropout 为0时，即什么也都没有操作
print('model:', model)
train_all_loss, test_all_loss, train_ACC, test_ACC = train_and_test(model=model, epochs=20, lr=0.01)
print('-----', train_all_loss, test_all_loss, train_ACC, test_ACC)


model_2 = MyTenNet(dropout=0.3)  # 模型
print('model:', model)
train_all_loss_2, test_all_loss_2, train_ACC_2, test_ACC_2 = train_and_test(model=model_2, epochs=20, lr=0.01)
print('-----', train_all_loss_2, test_all_loss_2, train_ACC_2, test_ACC_2)

model_3 = MyTenNet(dropout=0.6)  # 模型
print('model:', model)
train_all_loss_3, test_all_loss_3, train_ACC_3, test_ACC_3 = train_and_test(model=model_3, epochs=20, lr=0.01)
print('-----', train_all_loss_3, test_all_loss_3, train_ACC_3, test_ACC_3)

model_4 = MyTenNet(dropout=0.9)  # 模型
print('model:', model)
train_all_loss_4, test_all_loss_4, train_ACC_4, test_ACC_4 = train_and_test(model=model_4, epochs=20, lr=0.01)
print('-----', train_all_loss_4, test_all_loss_4, train_ACC_4, test_ACC_4)

# 完成loss的显示
drop_name_1 = ['dropout=0', 'dropout=0.3', 'dropout=0.6', 'dropout=0.9', '手动实现不同的dropout-Loss变化']
drop_train_1 = [train_all_loss, train_all_loss_2, train_all_loss_3, train_all_loss_4]
drop_test_1 = [test_all_loss, test_all_loss_2, test_all_loss_3, test_all_loss_4]
picture(drop_name_1, drop_train_1, drop_test_1)

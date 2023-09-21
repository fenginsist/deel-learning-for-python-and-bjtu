# torch 实现 FNN 解决 二分类问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

import torch

# 暂时使用这个
from homework_10_FNN_ten_torch_utils import MyTenNet, train_and_test_torch

"""
实现 rmsprop 优化器

Fashion-MNIST数据集:
    该数据集包含60,000个用于训练的图像样本和10,000个用于测试的图像样本。
    图像是固定大小(28x28像素)，其值为0到1。为每个图像都被平展并转换为784（28*28）。
"""





model = MyTenNet(dropout=0)  # 模型
RMSprop_optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9, eps=1e-6)
print('model:', model)
train_all_loss, test_all_loss, train_ACC, test_ACC = train_and_test_torch(model=model, epochs=20, optimizer=RMSprop_optimizer)
print('-----', train_all_loss, test_all_loss, train_ACC, test_ACC)

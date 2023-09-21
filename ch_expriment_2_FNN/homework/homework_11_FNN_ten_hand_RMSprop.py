# 手动 实现 FNN 解决 二分类问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

import torch

# 暂时使用这个
from homework_10_FNN_ten_hand_utils import MyTenNet, train_and_test

"""
实现 rmsprop 优化器

Fashion-MNIST数据集:
    该数据集包含60,000个用于训练的图像样本和10,000个用于测试的图像样本。
    图像是固定大小(28x28像素)，其值为0到1。为每个图像都被平展并转换为784（28*28）。
"""


# 初始化
def init_rmsprop(params):
    s_w1, s_b1, s_w2, s_b2 = torch.zeros(params[0].shape), torch.zeros(params[1].shape), \
        torch.zeros(params[2].shape), torch.zeros(params[3].shape)
    return (s_w1, s_b1, s_w2, s_b2)


# 对每一个参数进行RMSprop法
def rmsprop(params, states, lr=0.01, gamma=0.9):
    gamma, eps = gamma, 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= lr * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()


model = MyTenNet(dropout=0)  # 模型
print('model:', model)
train_all_loss, test_all_loss, train_ACC, test_ACC = train_and_test(model=model, epochs=20, init_states=init_rmsprop,
                                                                    optimizer=rmsprop)
print('-----', train_all_loss, test_all_loss, train_ACC, test_ACC)

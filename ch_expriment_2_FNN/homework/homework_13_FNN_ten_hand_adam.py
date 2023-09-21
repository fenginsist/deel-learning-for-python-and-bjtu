# 手动 实现 FNN 解决 二分类问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

import torch

# 暂时使用这个
from homework_10_FNN_ten_hand_utils import MyTenNet, train_and_test

"""
实现 adam 优化器

Fashion-MNIST数据集:
    该数据集包含60,000个用于训练的图像样本和10,000个用于测试的图像样本。
    图像是固定大小(28x28像素)，其值为0到1。为每个图像都被平展并转换为784（28*28）。
"""


def init_adam_states(params):
    v_w1, v_b1, v_w2, v_b2 = torch.zeros(params[0].shape), torch.zeros(params[1].shape), \
        torch.zeros(params[2].shape), torch.zeros(params[3].shape)
    s_w1, s_b1, s_w2, s_b2 = torch.zeros(params[0].shape), torch.zeros(params[1].shape), \
        torch.zeros(params[2].shape), torch.zeros(params[3].shape)
    return ((v_w1, s_w1), (v_b1, s_b1), (v_w2, s_w2), (v_b2, s_b2))


# 根据Adam算法思想手动实现Adam
Adam_t = 0.01


def Adam(params, states, lr=0.01, t=Adam_t):
    global Adam_t
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * (p.grad ** 2)
            v_bias_corr = v / (1 - beta1 ** Adam_t)
            s_bias_corr = s / (1 - beta2 ** Adam_t)
        p.data -= lr * v_bias_corr / (torch.sqrt(s_bias_corr + eps))
    p.grad.data.zero_()
    Adam_t += 1


model = MyTenNet(dropout=0)  # 模型
print('model:', model)
train_all_loss, test_all_loss, train_ACC, test_ACC = train_and_test(model=model, epochs=20,
                                                                    init_states=init_adam_states,
                                                                    optimizer=Adam)
print('-----', train_all_loss, test_all_loss, train_ACC, test_ACC)

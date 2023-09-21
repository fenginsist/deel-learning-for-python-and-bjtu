import torch
import numpy as np

# 实验2中的一些 激活函数案例

print('------------------------------------手动实现sigmoid激活函数')
# 手动实现sigmoid激活函数
x = torch.randn(4)
print('x:', x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


y = sigmoid(x)
print(y)

print('------------------------------------调用pytorch实现的sigmoid激活函数')

y = torch.sigmoid(x)
print(y)

print('------------------------------------手动实现tanh激活函数')


def fun_tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


y = fun_tanh(x)
print(y)

print('------------------------------------调用pytorch实现的tanh激活函数')
y = torch.tanh(x)
print(y)

print('------------------------------------手动实现ReLU激活函数')


def fun_relu(x):
    x = np.where(x >= 0, x, 0)
    return torch.tensor(x)


y = fun_relu(x)
print(y)

print('------------------------------------调用pytorch实现的ReLU激活函数')
y = torch.nn.functional.relu(x)
print(y)
print('------------------------------------手动实现leakyReLU激活函数')


def fun_leakrelu(x, gamma):
    x = np.where(x > 0, x, x * gamma)
    return torch.tensor(x)


y = fun_leakrelu(x, 0.2)
print(y)
print('------------------------------------调用pytorch实现的leakyReLU激活函数')
y = torch.nn.functional.leaky_relu(x, 0.2)
print(y)

print('------------------------------------手动实现eLU激活函数')


def fun_elu(x, gamma):
    x = np.where(x > 0, x, gamma * (np.exp(x) - 1))
    return torch.tensor(x)


y = fun_elu(x, 0.2)
print(y)
print('------------------------------------调用pytorch实现的eLU激活函数')
y = torch.nn.functional.elu(x, 0.2)
print(y)

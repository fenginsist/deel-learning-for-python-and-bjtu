import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果有gpu则在gpu上计算 加快计算速度
print(f'当前使用的device为{device}')

# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

print('-----------------------------手动生成回归任务的数据集，要求:-----------------------------------')
"""
一、手动生成回归任务的数据集，要求:
生成单个数据集。
1 数据集的大小为10000且训练集大小为7000，测试集大小为3000。
2 数据集的样本特征维度p为 500，且服从如下的高维线性函数:
"""
true_w, true_b = torch.ones(500, 1) * 0.0056, 0.028  # 真实的权重w、真实的偏差c

# 训练集
train_feature_line = torch.tensor(np.random.normal(0, 1, (7000, 500)), dtype=torch.float, requires_grad=True)
train_labels_line = torch.matmul(train_feature_line, true_w) + true_b  # matmul：矩阵维度相同，对应相乘。
train_labels_line += torch.tensor(np.random.normal(0, 0.01, size=train_labels_line.size()), dtype=torch.float,
                                  requires_grad=True)
print(train_feature_line.shape)
print(train_labels_line.shape)

# 测试集
test_feature_line = torch.tensor(np.random.normal(0, 1, (3000, 500)), dtype=torch.float, requires_grad=True)
test_labels_line = torch.matmul(test_feature_line, true_w) + true_b  # matmul：矩阵维度相同，对应相乘。
test_labels_line += torch.tensor(np.random.normal(0, 0.01, size=test_labels_line.size()), dtype=torch.float,
                                 requires_grad=True)
print(test_feature_line.shape)
print(test_labels_line.shape)

print('-----------------------------手动生成二分类任务的数据集，要求:-----------------------------------')
"""二、手动生成二分类任务的数据集，要求:
共生成两个数据集。
1 两个数据集的大小均为10000且训练集大小为7000，测试集大小为3000。
2 两个数据集的样本特征x的维度均为200，且分别服从均值互为相反数且方差相同的正态分布。
3 两个数据集的样本标签分别为0和1。
"""
train_feature_2sort = torch.normal(mean=1, std=0.01, size=(7000, 200), dtype=torch.float, requires_grad=True)
print('train_feature_2sort.shape: ', train_feature_2sort.shape)
# print(train_feature_2sort)
train_labels_2sort = torch.zeros(7000, requires_grad=True)
print('train_labels_2sort.shape: ', train_labels_2sort.shape)
# print(train_labels_2sort)

test_feature_2sort = torch.normal(mean=-1, std=0.01, size=(3000, 200), dtype=torch.float, requires_grad=True)
print('test_feature_2sort.shape: ', test_feature_2sort.shape)
# print(test_feature_2sort)
test_labels_2sort = torch.zeros(3000, requires_grad=True)
print('test_labels_2sort.shape: ', test_labels_2sort.shape)
# print(test_labels_2sort)

print('-----------------------------FashionMNIST 手写体数据集介绍:-----------------------------------')
"""
三、MNIST手写体数据集介绍:
该数据集包含60,000个用于训练的图像样本和10,000个用于测试的图像样本。
l 图像是固定大小(28x28像素)，其值为0到1。为每个图像都被平展并转换为784(28 * 28)个特征的一维 numpy数组。
"""

# 下载 MNIST 手写数字数据集
train_dataset = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST/train", train=True,
                                                  transform=transforms.ToTensor(),
                                                  download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST/test", train=False,
                                                 transform=transforms.ToTensor(),
                                                 download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

for X, y in train_loader:
    print(X.shape, y.shape)
    break
for X, y in test_loader:
    print(X.shape, y.shape)
    break


# 绘制图像的代码
def picture(name, trainl, testl, type='Loss'):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.title(name)  # 命名
    plt.plot(trainl, c='g', label='Train ' + type)
    plt.plot(testl, c='r', label='Test ' + type)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


# print(f'回归数据集   样本总数量{len(traindataset1) + len(testdataset1)},训练样本数量{len(traindataset1)},测试样本数量{len(testdataset1)}')
# print(f'二分类数据集 样本总数量{len(traindataset2) + len(testdataset2)},训练样本数量{len(traindataset2)},测试样本数量{len(testdataset2)}')
# print(f'多分类数据集 样本总数量{len(traindataset3) + len(testdataset3)},训练样本数量{len(traindataset3)},测试样本数量{len(testdataset3)}')

# 绘制图像的代码
def picture(name, trainl, testl, xlabel='Epoch', ylabel='Loss'):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.figure(figsize=(8, 3))
    plt.title(name[-1])  # 命名
    color = ['g', 'r', 'b', 'c']
    if trainl is not None:
        plt.subplot(121)
        for i in range(len(name) - 1):
            plt.plot(trainl[i], c=color[i], label=name[i])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
    if testl is not None:
        plt.subplot(122)
        for i in range(len(name) - 1):
            plt.plot(testl[i], c=color[i], label=name[i])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()

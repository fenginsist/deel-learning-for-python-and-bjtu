# 要求动手从0实现softmax回归(只借助Tensor和Numpy相关的库)在Fashion-MNIST数据 集上进行训练和测试，
# 并从loss、训练集以及测试集上的准确率等多个角度对结果进行分析 (要求从零实现交叉熵损失函数)
# https://blog.csdn.net/qq_37534947/article/details/108179408

# 多分类问题，采用 softmax 函数，采用 交叉熵损失函数，梯度下降 做参数的优化


# 注意的点
# 1、loss记得 .sum()

# 问题：
# 1、这不是多分类吗？怎么才能看出来效果呢
# 2、每个样本的数据格式为:28*28*1(高*宽*通道)，该怎么理解呢                                   ：长28个像素，宽28个像素。
# 3、这里权重和偏差 是 784*10，why, 逻辑回归是 一个向量，为啥这里那么大呢？
# 4、打断点时，pre_y 是256*10，然后里面有很多 H、T 这是啥呢？data 是模型计算出来的数据吗？很困惑   ：转置
# 5、gather 函数


import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

# 下载的数据
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False,
                                               transform=transforms.ToTensor())

# 读取数据
batch_size = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

# 输入与输出(也是矩阵的行和列) ??????
input_num = 784
output_num = 10

# 初始化权重和偏差
w = torch.tensor(np.random.normal(0, 0.01, (input_num, output_num)), dtype=torch.float, requires_grad=True)
b = torch.zeros(output_num, dtype=torch.float, requires_grad=True)
print('w:', w)
print('w.size():', w.size())  # torch.Size([784, 10])
print('b:', b)
print('b.size():', b.size())  # torch.Size([10]) 一个行向量列


# 初始化模型
def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


# 还是先计算 线性判别函数，再计算非线性决策函数
def softmax_net(x):
    # 重点：先是 1、线性判别函数，2、然后在是softmax 非线性决策函数
    # 线性判别函数，记得使用 view(-1， int) 使其可以进行广播
    f_x = torch.mm(x.view((-1, input_num)), w) + b
    return softmax(f_x)


# 损失函数：手动实现交叉熵损失函数
def cross_entropy(pre_y, y):
    return -torch.log(pre_y.gather(1, y.view(-1, 1)))  # y.view(-1, 1)：变成了 256行 1列的数据。


# 优化器
# 手动定义sgd优化函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        # print(len(X)) 小批量数据集 每个X中有 256个图像
        # print((net(X).argmax(dim=1)==y).float().sum().item())
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 迭代
epoch = 10
lr = 1e-4

net = softmax_net
loss = cross_entropy

for i in range(epoch):
    train_l_num, train_acc_num, n = 0.0, 0.0, 0
    for X, y in train_iter:  # X 为小批量256个图像 1*28*28 y为标签
        # 1、训练模型，得倒预测值
        pre_y = net(X)      # 这里没有传 w和b，因为模型里面引用的是全局的 权重和偏差
        # 2、损失函数，得出损失
        l = loss(pre_y, y).sum()
        # 3、反向传播
        l.backward()
        # 4、优化参数
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
        # 4、梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
        # 5、计算每个epoch的loss
        train_l_num += l.item()

        # # 计算训练样本的准确率
        train_acc_num += (pre_y.argmax(dim=1) == y).sum().item()

        # 每一个epoch的所有样本数
        n += y.shape[0]
    test_acc = evaluate_accuracy(test_iter, net)
    print('epoch %d, loss %.4f， train_acc %.3f, test_acc %.3f' % (i + 1, train_l_num / n, train_acc_num / n, test_acc))

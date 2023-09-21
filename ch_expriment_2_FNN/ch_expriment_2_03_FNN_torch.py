# 实验2中 Torch.nn 实现 FNN 的多分类


import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 问题
# 1、输出为 10 个类别时，为啥没有看到使用 softmax 函数进行 非线性决策呢？

# 1)定义数据迭代器
# 下载的数据
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False,
                                               transform=transforms.ToTensor())
# 读取数据
# batch_size 每次批量加载 256个样本
batch_size = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)


# 辅助函数，绘图
def draw_loss(train_loss, test_loss):
    x = np.linspace(0, len(train_loss), len(train_loss))
    plt.plot(x, train_loss, label="Train Loss", linewidth=1.5)
    plt.plot(x, test_loss, label="Test Loass", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# 2&3)定义模型及其前向传播过程:一层神经网络 一层隐藏层

# 2) 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义模型参数
        num_inputs, num_outputs, num_hiddens = 784, 10, 256
        # 定义模型结构
        self.input_layer = lambda x: x.view(x.shape[0], -1)
        self.hidden_layer = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),                 # 疑问
            nn.ReLU()
        )
        self.output_layer = nn.Linear(num_hiddens, num_outputs)

        # 初始化参数
        for h_param in self.hidden_layer.parameters():
            torch.nn.init.normal_(h_param, mean=0, std=0.01)
        for o_param in self.output_layer.parameters():
            torch.nn.init.normal_(o_param, mean=0, std=0.01)

    # 3) 定义模型前向传播过程
    def forward(self, x):
        flatten_input = self.input_layer(x)
        hidden_output = self.hidden_layer(flatten_input)
        final_output = self.output_layer(hidden_output)
        return final_output


# 4)定义损失函数
loss_torch = nn.CrossEntropyLoss()
# 5)优化算法
SGD_torch = torch.optim.SGD


# 测试集准确率
def evaluate_accuracy(data_iter, model, loss_func):
    acc_sum, test_l_sum, n, c = 0.0, 0.0, 0, 0
    for X, y in data_iter:
        result = model.forward(X)
        acc_sum += (result.argmax(dim=1) == y).float().sum().item()
        test_l_sum += loss_func(result, y).item()
        n += y.shape[0]
        c += 1
    return acc_sum / n, test_l_sum / c


def train(net, train_iter, loss_func, num_epochs, lr=None, optimizer=None):
    train_loss_list = []
    test_loss_list = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, c = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            pre_y = net(X)
            l = loss_func(pre_y, y)
            # 梯度清零
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (pre_y.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            c += 1
        test_acc, test_loss = evaluate_accuracy(test_iter, net, loss_func)
        train_loss_list.append(train_l_sum / c)
        test_loss_list.append(test_loss)
        print('epoch %d, train loss %.4f, test_loss %.4f, train_acc %.3f, test_acc %.3f' % (epoch + 1, train_l_sum / c,
                                                                                            test_loss,
                                                                                            train_acc_sum / n,
                                                                                            test_acc))
    return train_loss_list, test_loss_list


# 训练前的准备工作
net = Net()
num_epochs = 20
lr = 0.01
optimizer = SGD_torch(net.parameters(), lr)
loss = loss_torch

# 开始训练
print("开始训练啦~~~~")
train_loss, test_loss = train(net, train_iter, loss, num_epochs, lr, optimizer)

# 训练完后绘制损失函数
draw_loss(train_loss, test_loss)

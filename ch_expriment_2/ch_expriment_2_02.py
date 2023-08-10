# 实验2中 手动实现前馈神经网络


import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1)定义数据迭代器
# 下载的数据
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False,
                                               transform=transforms.ToTensor())
# 读取数据
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
class Net:
    def __init__(self):
        # 定义并初始化模型参数
        num_inputs, num_outputs, num_hiddens = 784, 10, 256
        # 以下是一层神经网络（隐藏层）的隐藏层和输出层的权重和偏差。
        w1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float, requires_grad=True)
        b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
        w2 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float, requires_grad=True)
        b2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)

        # 告知PvTorch框架，上诸四个李量需求梯障,
        # 我上面加了  requires_grad=True 这个属性，所以下面可以注释掉了
        self.params = [w1, b1, w2, b2]
        # for param in self.params:
        #     param.requires_grad = True

        # 定义模型结构 ?????: 定义激活函数
        self.input_layer = lambda x: x.view(x.shape[0], -1)
        self.hidden_layer = lambda x: self.my_ReLu(torch.matmul(x, w1.t()) + b1)
        self.output_layer = lambda x: torch.matmul(x, w2.t()) + b2

    def my_ReLu(self, x):
        return torch.max(input=x, other=torch.tensor(0, 0))

    # 3) 定义模型前向传播过程
    def forward(self, x):
        flatten_input = self.input_layer(x)
        hidden_output = self.hidden_layer(flatten_input)
        final_output = self.output_layer(hidden_output)
        return final_output


model = Net()
print(model)


# 4)定义损失函数
def my_loss_entropy_loss(y_hat, labels):
    def log_softmax(y_hat):
        max_v = torch.max(y_hat, dim=1).values.unsqueeze(dim=1)
        return y_hat - max_v - torch.log(torch.exp(y_hat - max_v).sum(dim=1)).unsqueeze(dim=1)

    return (-log_softmax(y_hat))[range(len(y_hat)), labels].mean()


# 5)优化算法
def SGD(params, lr):
    for param in params:
        param.data -= lr * param.grad


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
            pre_y = net.forward(X)
            l = loss_func(pre_y, y)
            l.backward()
            optimizer(net.params, lr)
            for param in net.params:
                param.grad.data.zero_()
            #
            train_l_sum += l.item()
            #
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
optimizer = SGD
loss = my_loss_entropy_loss

# 开始训练
print("开始训练啦~~~~")
train_loss, test_loss = train(net, train_iter, loss, num_epochs, lr, optimizer)
# 训练完后绘制损失函数
draw_loss(train_loss, test_loss)

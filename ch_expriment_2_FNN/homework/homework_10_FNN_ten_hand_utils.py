import torch
import torch.nn as nn
import numpy as np
import random
import time
import torchvision
import torchvision.transforms as transforms

"""
手动实现 FNN 解决 多分类问题 的工具类，下面的文件都基于此做实验，进行测试 momentum、rmsprop、adam 三种优化方法
手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278
"""


# 训练集
batch_size = 128
train_dataset = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST/train", train=True,
                                                  transform=transforms.ToTensor(),
                                                  download=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 测试集
test_dataset = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST/test", train=False,
                                                 transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 2&3)定义模型及其前向传播过程:一层神经网络 一层隐藏层
# 2) 定义模型
class MyTenNet():
    def __init__(self, dropout=0.0):
        # global dropout
        self.dropout = dropout              # 当 dropout 为0时，即什么也都没有操作
        print('dropout: ', self.dropout)
        self.is_train = None

        # 设置输入层（特征数）、隐藏层（神经元个数）、输出层的节点数
        num_inputs, num_hiddens, num_outputs = 28 * 28, 256, 10  # 输入层、隐藏层、输出层 的 神经元个数 # 十分类问题
        # 以下是一层神经网络（隐藏层）的隐藏层和输出层的权重和偏差。
        w1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float, requires_grad=True)
        b1 = torch.zeros(num_hiddens, dtype=torch.float32, requires_grad=True)
        w2 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float, requires_grad=True)
        b2 = torch.zeros(num_outputs, dtype=torch.float32, requires_grad=True)
        self.params = [w1, b1, w2, b2]

        # 定义模型结构 *****: 定义激活函数，输入层变成一个一维向量、隐藏层激活函数 relu 函数、输出层激活函数
        self.input_layer = lambda x: x.view(x.shape[0], -1)  # 模型输入，拉平成向量
        self.hidden_layer = lambda x: self.my_relu(torch.matmul(x, w1.t()) + b1)  # 隐藏层，使用 relu 函数。
        self.output_layer = lambda x: torch.matmul(x, w2.t()) + b2  # 模型输出

    def my_relu(self, x):
        return torch.max(input=x, other=torch.tensor(0.0))

    # 以下两个函数分别在训练和测试前调用，选择是否需要dropout
    def train(self):
        self.is_train = True

    def test(self):
        self.is_train = False

    """
        定义dropout层
        x: 输入数据
        dropout: 随机丢弃的概率
        """

    def dropout_layer(self, x):
        dropout = self.dropout
        assert 0 <= dropout <= 1  # dropout值必须在0-1之间
        # dropout==1，所有元素都被丢弃。
        if dropout == 1:
            return torch.zeros_like(x)
            # 在本情况中，所有元素都被保留。
        if dropout == 0:
            return x
        mask = (torch.rand(x.shape) < 1.0 - dropout).float()  # rand()返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数
        return mask * x / (1.0 - dropout)

    # 3) 定义模型前向传播过程
    # 多分类，为啥没有看到使用 softmax 非线性决策函数呢。答案在下面
    def forward(self, x):
        flatten_input = self.input_layer(x)
        if self.is_train:  # 如果是训练过程，则需要开启dropout 否则 需要关闭 dropout
            flatten_input = self.dropout_layer(flatten_input)
        hidden_output = self.hidden_layer(flatten_input)
        if self.is_train:
            hidden_output = self.dropout_layer(hidden_output)
        final_output = self.output_layer(hidden_output)
        """
        值得注意的是，分开定义 𝐬𝐨𝐟𝐭𝐦𝐚𝐱 运算和交叉熵损失函数可能会造成数值不稳定。
        因此，PyTorch提供了一个包括 𝐬𝐨𝐟𝐭𝐦𝐚𝐱 运算和交叉熵损失计算的函数。它的数值稳 定性更好。
        所以在输出层，我们不需要再进行 𝐬𝐨𝐟𝐭𝐦𝐚𝐱 操作
        """
        return final_output


# 损失函数:交叉熵损失函数
def cross_entropy(y_hat, y):
    # return -torch.log(y_hat.gather(1, y.view(-1, 1)))
    return torch.mean(-torch.log(y_hat.gather(1, y.view(-1, 1))))


# 小批量梯度下降 优化算法
def mySGD(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 定义L2范数惩罚项 参数 w 为模型的 w 在本次实验中为[w_1, w_2] batch_size=128
def l2_penalty(w):
    cost = 0
    for i in range(len(w)):
        cost += (w[i] ** 2).sum()
    return cost / batch_size / 2


"""
定义训练函数
model:定义的模型 默认为MyNet(0) 即无dropout的初始网络

init_states：是 optimizer 优化函数的初始化
optimizer:定义的优化函数，默认为自己定义的mySGD函数

epochs:训练总轮数 默认为30
lr :学习率 默认为0.1

L2：是否需要 L2正则化 
lambd：L2正则化的 惩罚系数
"""


def train_and_test(model=MyTenNet(), init_states=None, optimizer=mySGD, epochs=20, lr=0.01, L2=False, lambd=0):
    train_all_loss = []  # 记录训练集上得loss变化
    test_all_loss = []  # 记录测试集上的loss变化
    train_ACC, test_ACC = [], []  # 记录正确的个数
    begin_time = time.time()
    # 激活函数为自己定义的mySGD函数
    # criterion = cross_entropy # 损失函数为交叉熵函数
    loss = nn.CrossEntropyLoss()  # 损失函数

    """表明当前处于训练状态，允许使用dropout"""
    model.train()

    for epoch in range(epochs):
        train_l, train_acc_num = 0, 0
        for data, label in train_loader:
            pre = model.forward(data)
            l = loss(pre, label).sum()  # 计算每次的损失值
            # 若L2为True则表示需要添加L2范数惩罚项
            if L2 == True:
                l += lambd * l2_penalty(model.w)

            train_l += l.item()
            l.backward()  # 反向传播
            # 若当前states为 None表示 使用的是 默认的优化函数mySGD
            if init_states == None:
                optimizer(model.params, lr, 128)  # 使用小批量随机梯度下降迭代模型参数
            # 否则的话使用的是自己定义的优化器，通过传入的参数，来实现优化效果
            else:
                states = init_states(model.params)
                optimizer(model.params, states, lr=lr)
            # 梯度清零
            train_acc_num += (pre.argmax(dim=1) == label).sum().item()
            for param in model.params:
                param.grad.data.zero_()

            # print(train_each_loss)
        train_all_loss.append(train_l)  # 添加损失值到列表中
        train_ACC.append(train_acc_num / len(train_dataset))  # 添加准确率到列表中
        # print('-----------------------开始测试啦')
        with torch.no_grad():
            is_train = False  # 表明当前为测试阶段，不需要dropout参与
            test_loss, test_acc_num = 0, 0
            for X, y in test_loader:
                p = model.forward(X)  # 去掉了 .sum()
                ll = loss(p, y)
                test_loss += ll.item()
                test_acc_num += (p.argmax(dim=1) == y).sum().item()
            test_all_loss.append(test_loss)
            test_ACC.append(test_acc_num / len(test_dataset))  # # 添加准确率到列表中
        if epoch == 0 or (epoch + 1) % 4 == 0:
            print('epoch: %d | train loss:%.5f | test loss:%.5f | train acc: %.2f | test acc: %.2f'
                  % (epoch + 1, train_l, test_loss, train_ACC[-1], test_ACC[-1]))
    end_time = time.time()
    print("homework_10_FNN_ten_hand_utils.py:手动实现前馈网络-多分类实验 %d轮 总用时: %.3fs" % (epochs, end_time - begin_time))
    return train_all_loss, test_all_loss, train_ACC, test_ACC
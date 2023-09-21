# 手动实现 FNN 解决 多分类问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

import torch
import numpy as np
import random
import time
import torchvision
import torchvision.transforms as transforms

# RuntimeError: grad can be implicitly created only for scalar outputs
# 该错误是：只能隐式地为标量输出创建Grad。
# 该错误研究了好久，发现损失函数返回的是向量，应该是标量，所以加上torch.mean() 即可解决此问题。

# 训练集
train_dataset = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST/train", train=True,
                                                  transform=transforms.ToTensor(),
                                                  download=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
# 测试集
test_dataset = torchvision.datasets.FashionMNIST(root="./Datasets/FashionMNIST/test", train=False,
                                                 transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)


# 2&3)定义模型及其前向传播过程:一层神经网络 一层隐藏层
# 2) 定义模型
class MyTenNet():
    def __init__(self):
        # 设置输入层（特征数）、隐藏层（神经元个数）、输出层的节点数
        num_inputs, num_hiddens, num_outputs = 28 * 28, 256, 10  # 输入层、隐藏层、输出层 的 神经元个数
        # 以下是一层神经网络（隐藏层）的隐藏层和输出层的权重和偏差。
        w1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float, requires_grad=True)
        b1 = torch.zeros(num_hiddens, dtype=torch.float32, requires_grad=True)
        w2 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float, requires_grad=True)
        b2 = torch.zeros(num_outputs, dtype=torch.float32, requires_grad=True)
        self.params = [w1, b1, w2, b2]

        # 定义模型结构 *****: 定义激活函数，输入层变成一个一维向量、隐藏层激活函数 relu 函数、输出层激活函数
        self.input_layer = lambda x: x.view(x.shape[0], -1)  # 模型输入，拉平成向量
        self.hidden_layer = lambda x: self.my_ReLu(torch.matmul(x, w1.t()) + b1)  # 隐藏层，使用 relu 函数。
        self.output_layer = lambda x: torch.matmul(x, w2.t()) + b2  # 模型输出

    def my_ReLu(self, x):
        return torch.max(input=x, other=torch.tensor(0.0))

    # 3) 定义模型前向传播过程
    # 多分类，为啥没有看到使用 softmax 非线性决策函数呢。答案在下面
    def forward(self, x):
        flatten_input = self.input_layer(x)
        hidden_output = self.hidden_layer(flatten_input)
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


# 优化算法
def mySGD(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


model = MyTenNet()
loss = cross_entropy  #
sgd = mySGD
lr = 0.15
batch_size = 128  # 和上面的数据集的 batch 保持一致。
epochs = 40
train_all_loss = []  # 记录训练集上得loss变化
test_all_loss = []  # 记录测试集上的loss变化
train_ACC, test_ACC = [], []  # 记录正确的个数
begin_time = time.time()
print('------------------开始训练啦')
for epoch in range(epochs):
    train_l, train_acc_num = 0, 0
    for data, label in train_loader:
        pre = model.forward(data)
        l = loss(pre, label).sum()  # 计算每次的损失值
        train_l += l.item()
        l.backward()  # 反向传播
        sgd(model.params, lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
        # 梯度清零
        train_acc_num += (pre.argmax(dim=1) == label).sum().item()
        for param in model.params:
            param.grad.data.zero_()

        # print(train_each_loss)
    train_all_loss.append(train_l)  # 添加损失值到列表中
    train_ACC.append(train_acc_num / len(train_dataset))  # 添加准确率到列表中
    # print('-----------------------开始测试啦')
    with torch.no_grad():
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
print("手动实现前馈网络-多分类实验 %d轮 总用时: %.3fs" % (epochs, begin_time - end_time))

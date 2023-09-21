

import torch
import torch.nn as nn
import torch.optim as optim
import time
import torchvision
import torchvision.transforms as transforms

"""
torch 实现 FNN 解决 多分类问题 的工具类，下面的文件都基于此做实验，进行测试 momentum、rmsprop、adam 三种优化方法
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
class MyTenNet(nn.Module):
    def __init__(self, dropout=0.0):
        super(MyTenNet, self).__init__()
        # global dropout
        # self.dropout = dropout
        # print('dropout: ', self.dropout)
        # self.is_train = None

        # 设置输入层（特征数）、隐藏层（神经元个数）、输出层的节点数
        num_inputs, num_hiddens, num_outputs = 28 * 28, 256, 10  # 输入层、隐藏层、输出层 的 神经元个数 # 十分类问题
        # 定义模型结构 *****: 定义激活函数，输入层变成一个一维向量、隐藏层激活函数 relu 函数、输出层激活函数
        self.input_layer = nn.Flatten()  # 模型输入，拉平成向量
        self.hidden_layer = nn.Linear(num_inputs, num_hiddens)  # 隐藏层, 使用 relu 函数。
        # 根据设置的dropout设置丢失率
        self.drop = nn.Dropout(dropout)
        self.output_layer = nn.Linear(num_hiddens, num_outputs)  # 模型输出
        # 使用relu作为激活函数
        self.relu = nn.ReLU()

    # 以下两个函数分别在训练和测试前调用，选择是否需要dropout
    # def train(self):
    #     self.is_train = True

    # def test(self):
    #     self.is_train = False

    # 3) 定义模型前向传播过程
    # 多分类，为啥没有看到使用 softmax 非线性决策函数呢。答案在下面
    def forward(self, x):
        flatten_input = self.drop(self.input_layer(x))
        hidden_output = self.drop(self.hidden_layer(flatten_input))
        hidden_output = self.relu(hidden_output)
        final_output = self.output_layer(hidden_output)
        """
        值得注意的是，分开定义 𝐬𝐨𝐟𝐭𝐦𝐚𝐱 运算和交叉熵损失函数可能会造成数值不稳定。
        因此，PyTorch提供了一个包括 𝐬𝐨𝐟𝐭𝐦𝐚𝐱 运算和交叉熵损失计算的函数。它的数值稳 定性更好。
        所以在输出层，我们不需要再进行 𝐬𝐨𝐟𝐭𝐦𝐚𝐱 操作
        """
        return final_output


"""
定义训练函数
model:定义的模型 默认为MyNet(0) 即无dropout的初始网络
epochs:训练总轮数 默认为30
criterion:定义的损失函数，默认为cross_entropy
lr :学习率 默认为0.1
optimizer:定义的优化函数，默认为自己定义的mySGD函数
"""


def train_and_test_torch(model=MyTenNet(), optimizer=None, epochs=20, lr=0.01, weight_decay=0.0):
    # 优化函数, 默认情况下weight_decay为0 通过更改 weight_decay 的值可以实现L2正则化。
    # 默认的优化函数为SGD 可以根据参数来修改优化函数
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_all_loss = []  # 记录训练集上得loss变化
    test_all_loss = []  # 记录测试集上的loss变化
    train_ACC, test_ACC = [], []  # 记录正确的个数
    begin_time = time.time()
    # 激活函数为自己定义的mySGD函数
    loss = nn.CrossEntropyLoss()  # 交叉熵函数损失函数 # criterion = cross_entropy # 损失函数为交叉熵函数，这是自己实现的
    model.train()  # 表明当前处于训练状态，允许使用dropout

    for epoch in range(epochs):
        train_l, train_acc_num = 0, 0
        for data, label in train_loader:
            pre = model.forward(data)
            l = loss(pre, label).sum()  # 计算每次的损失值
            # 若L2为True则表示需要添加L2范数惩罚项, 这是手动添加的方式，torch方式直接 在优化函数里进行设置
            # if L2 == True:
            #     l += lambd * l2_penalty(model.w)
            optimizer.zero_grad()  # 梯度清零
            l.backward()  # 反向传播
            optimizer.step()  # 梯度更新参数

            train_l += l.item()
            train_acc_num += (pre.argmax(dim=1) == label).sum().item()
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
    print("homework_10_FNN_ten_torch_utils.py: torch 实现前馈网络-多分类实验 %d轮 总用时: %.3fs" % (epochs, end_time - begin_time))
    return train_all_loss, test_all_loss, train_ACC, test_ACC


model = MyTenNet(dropout=0)
train_all_loss, test_all_loss, train_ACC, test_ACC = train_and_test_torch(model=model, epochs=20, lr=0.01)
print('--------------:', train_all_loss, test_all_loss, train_ACC, test_ACC)
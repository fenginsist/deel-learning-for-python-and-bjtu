import torch
from ch_expriment_0_01 import MyLayer  # 自定义层

N, D_in, D_out = 10, 5, 3  # 一共10组样本，输入特征为5，输出特征为3


# 1、先定义一个模型
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 第一句话，调用父类的构造函数
        self.mylayer1 = MyLayer(D_in, D_out)

    # def __call__
    def forward(self, x):
        x = self.mylayer1(x)
        return x


model = MyNet()
print(model)

# 2、准备数据
# 创建输入、输出数据
x = torch.randn(N, D_in)  # (10，5)
y = torch.randn(N, D_out)  # (10，3)

# 3、定义损失函数 与 优化器
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(10):
    # 第一步:数据的前向传播，计算预测值p_pred，              重点：执行的是模型层的forward函数
    y_pred = model(x)  # 默认会执行 forward函数 进行前向传播
    # 第二步:计算计算预测值p_pred与真实值的误差
    loss = loss_fn(y_pred, y)
    print(f"第{i}个epoch，损失是{loss.item()}")
    # 在反向传播之前，将模型的梯度归零
    optimizer.zero_grad()
    # 第三步:反向传播误差
    loss.backward()
    # 直接通过梯度一步到位，更新完整个网络的训练参数
    optimizer.step()

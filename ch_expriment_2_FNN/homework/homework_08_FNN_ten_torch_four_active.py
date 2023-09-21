# torch 实现 FNN 解决 二分类问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

from homework_07_FNN_ten_torch import MyTenNet
from homework_07_FNN_ten_torch import train_and_test_fun
from homework_07_FNN_ten_torch import ComPlot
import matplotlib.pyplot as plt

"""
在上述实现的多分类任务中使用至少三种不同的激活函数，本次实验使用分别使用四个激活函数：Relu、Tanh、Sigmoid、ELU
"""

model = MyTenNet()  # 默认，激活函数是 relu
train_all_loss_relu, test_all_loss_relu, train_ACC_relu, test_ACC_relu = train_and_test_fun(model=model)
print('-------------', train_all_loss_relu, test_all_loss_relu, train_ACC_relu, test_ACC_relu)

model = MyTenNet(act='tanh')
train_all_loss_tanh, test_all_loss_tanh, train_ACC_tanh, test_ACC_tanh = train_and_test_fun(model=model)
print('-------------', train_all_loss_tanh, test_all_loss_tanh, train_ACC_tanh, test_ACC_tanh)

model = MyTenNet(act='sigmoid')
train_all_loss_sigmoid, test_all_loss_sigmoid, train_ACC_sigmoid, test_ACC_sigmoid = train_and_test_fun(model=model)
print('-------------', train_all_loss_sigmoid, test_all_loss_sigmoid, train_ACC_sigmoid, test_ACC_sigmoid)

model = MyTenNet(act='elu')
train_all_loss_elu, test_all_loss_elu, train_ACC_elu, test_ACC_elu = train_and_test_fun(model=model)
print('-------------', train_all_loss_elu, test_all_loss_elu, train_ACC_elu, test_ACC_elu)

plt.figure(figsize=(16, 3))
plt.subplot(141)
ComPlot([train_all_loss_relu, test_all_loss_relu, train_ACC_relu, test_ACC_relu], title='Train_Loss')
plt.subplot(142)
ComPlot([train_all_loss_tanh, test_all_loss_tanh, train_ACC_tanh, test_ACC_tanh], title='Test_Loss')
plt.subplot(143)
ComPlot([train_all_loss_sigmoid, test_all_loss_sigmoid, train_ACC_sigmoid, test_ACC_sigmoid], title='Train_ACC')
plt.subplot(144)
ComPlot([train_all_loss_elu, test_all_loss_elu, train_ACC_elu, test_ACC_elu], title='Test_ACC')
plt.show()

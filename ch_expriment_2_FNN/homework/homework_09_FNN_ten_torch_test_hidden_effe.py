# torch 实现 FNN 解决 二分类问题
# 手动以及使用torch.nn实现前馈神经网络实验：https://blog.csdn.net/m0_52910424/article/details/127819278

from homework_07_FNN_ten_torch import MyTenNet
from homework_07_FNN_ten_torch import train_and_test_fun
from homework_07_FNN_ten_torch import ComPlot
import matplotlib.pyplot as plt

"""
使用不同的隐藏层层数和隐藏单元个数，进行对比实验并分析实验结果，为了体现控制变量原则，统一使用RELU为激活函数
Model  Hidden layer num         each hidden num         rank
model           1               [128]                   2
model1          2               [512,256]               1
model2          3               [512,256,128]           3
model3          1               [256]                   4
"""

model = MyTenNet(1, 28 * 28, [128], 10, act='relu')  # 默认，激活函数是 relu
train_all_loss_relu, test_all_loss_relu, train_ACC_relu, test_ACC_relu = train_and_test_fun(model=model)
print('-------------', train_all_loss_relu, test_all_loss_relu, train_ACC_relu, test_ACC_relu)

model1 = MyTenNet(2, 28 * 28, [512, 256], 10, act='relu')  # 默认，激活函数是 relu
train_all_loss_relu_1, test_all_loss_relu_1, train_ACC_relu_1, test_ACC_relu_1 = train_and_test_fun(model=model1)
print('-------------', train_all_loss_relu_1, test_all_loss_relu_1, train_ACC_relu_1, test_ACC_relu_1)

model2 = MyTenNet(3, 28 * 28, [512, 256, 128], 10, act='relu')  # 默认，激活函数是 relu
train_all_loss_relu_2, test_all_loss_relu_2, train_ACC_relu_2, test_ACC_relu_2 = train_and_test_fun(model=model2)
print('-------------', train_all_loss_relu_2, test_all_loss_relu_2, train_ACC_relu_2, test_ACC_relu_2)

model3 = MyTenNet(1, 28 * 28, [256], 10, act='relu')  # 默认，激活函数是 relu
train_all_loss_relu_3, test_all_loss_relu_3, train_ACC_relu_3, test_ACC_relu_3 = train_and_test_fun(model=model)
print('-------------', train_all_loss_relu_3, test_all_loss_relu_3, train_ACC_relu_3, test_ACC_relu_3)


plt.figure(figsize=(16, 3))
plt.subplot(141)
ComPlot([train_all_loss_relu, train_all_loss_relu_1, train_all_loss_relu_2, train_all_loss_relu_3],title='Train_Loss',flag='hidden')
plt.subplot(142)
ComPlot([test_all_loss_relu, test_all_loss_relu_1, test_all_loss_relu_2, test_all_loss_relu_3],title='Test_Loss',flag='hidden')
plt.subplot(143)
ComPlot([train_ACC_relu, train_ACC_relu_1, train_ACC_relu_2, train_ACC_relu_3],title='Train_ACC',flag='hidden')
plt.subplot(144)
ComPlot([test_ACC_relu, test_ACC_relu_1, test_ACC_relu_2, test_ACC_relu_3],title='Test_ACC', flag='hidden')
plt.show()

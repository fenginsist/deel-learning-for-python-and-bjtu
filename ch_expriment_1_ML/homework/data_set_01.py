import torch
import matplotlib.pyplot as plt

# 随机生成实验所需要的二分类数据集 𝑥1 和 𝑥2 ，分别对应的标签 𝑦1 和 𝑦2 .

n_data = torch.ones(50, 2)          # 数据的基本形态
print('n_data:', n_data.shape)
x1 = torch.normal(2 * n_data, 1)    # shape=(50, 2)
print('x1:', x1.shape)
y1 = torch.zeros(50)                # 类型0 shape=(50)
print('y1:', y1.shape)

x2 = torch.normal(-2 * n_data, 1)   # shape=(50, 2)
print('x2:', x2.shape)
y2 = torch.ones(50)                 # 类型1 shape=(50)
print('y2:', y2.shape)
print('y2.size():', y2.size())      # torch.Size([50])

# 注意 features, labels 数据的数据形式一定要像下面一样 (torch.cat 是合并数据)
features = torch.cat((x1, x2), 0).type(torch.FloatTensor)
labels = torch.cat((y1, y2), 0).type(torch.FloatTensor)
# print('features:', features)
print('features.size():', features.size())  # torch.Size([100, 2])
# print('labels:', labels)
print('labels.size():', labels.size())  # torch.Size([100])


plt.scatter(features.data.numpy()[:, 0], features.data.numpy()[:, 1], c=labels.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()




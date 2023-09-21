# 测试 torch 和 numpy 包函数
import torch
import torch as th
import numpy as np

# Tensor的算术运算：https://blog.csdn.net/Thumb_/article/details/114288063
# https://pytorch.org/docs/1.2.0/torch.html#torch.log

print("------------------------ np.random.normal():返回一组符合高斯分布的概率密度随机数")
# np.random.normal()：返回一组符合高斯分布的概率密度随机数。
# 函数使用：https://blog.csdn.net/qq_45154565/article/details/115690864
x = np.random.normal(0, 1, (5, 2))  # 返回是符合正态分布的 5行2列 的一组随机数
print(x)
print(x.dtype)
print(x.size)
print(x.shape)

print("------------------------ th.tensor(): np.random.normal() 转成 tensor张量")
features = th.tensor(x, dtype=th.float)
print(features)
print(features.size())  # torch.Size([5, 2])
print(features.shape)  # torch.Size([5, 2])
print(features[:, 0])  # 取第一列
print(features[:, 1])  # 取第二列
len = len(features)
print("长度为：", len)

print("------------------------ range():一个范围的数")

x = range(3)
print(x)
# 以下三个均无此属性：AttributeError: 'range' object has no attribute 'size'
# print(x.dtype)
# print(x.size)
# print(x.shape)

for i in range(5):
    print(i)

print("------------------------ range(x, y, z):一个范围的数")
x = range(0, 5, 5)
print('range(x, y, z)', x)

print("------------------------ list(): python 的数据类型： 元组")
x = list(range(len))
print(x)
# print(x.dtype)    # 'list' object has no attribute 'dtype'

print("------------------------ yield 关键字使用: return 的好兄弟，套娃")
# Python中的yield用法： https://zhuanlan.zhihu.com/p/268605982

# def test_yield():
#     x = 1
#     y = 10
#     while x < y:
#         yield  x
#         x += 1
#
# example = test_yield()
# example
# print(example)


print("------------------------ 函数原型：torch.normal(mean, std, *, generator=None, out=None): 均值是mean，标准差是std")
# torch.normal()的用法：https://blog.csdn.net/qq_41898018/article/details/119244914

# x = torch.normal(20, 2)

x = torch.normal(mean=0., std=1., size=(2, 2))  # 从一个标准正态分布 N～(0,1)，提取一个2x2的矩阵
print(x)

print("------------------------ torch.normal(torch.ones(10, 2), 1): 第一个参数为tensor，第二个目前不知")
data = torch.ones(10, 2)
print(data)
x = torch.normal(data, 1)
print(x)
print(x.size())
# Tensor 的三个对象， rank、shape、type。目前好像只有两个可以用
print(x.shape)
# print(x.rank)
print(x.type)

print("------------------------ torch.cat 是合并数据,类似于 SQL 中的 union all")

x1 = torch.ones(5)
print(x1)
x2 = torch.zeros(5)
print(x2)
x = torch.cat((x1, x2), 0).type(torch.FloatTensor)
print(x)

print(
    "------------------------ torch.exp(tensor, *, out=None)：exp 是指指数函数， e 的 x 次方;每一个元素都进行 e 的x次方")
# https://pytorch.org/docs/1.2.0/torch.html#torch.exp
x = torch.ones(1, 2)
print(x)
x = torch.exp(x)
print(x)

print("-----------------")

x = torch.tensor([1, 2])
print(x)
x = torch.exp(x)
print(x)

print("------------------------ torch.log(tensor, )：以e为底的对数")
# https://pytorch.org/docs/1.2.0/torch.html#torch.log
x = torch.tensor([10, 10])
print(x)
x = torch.log(x)
print(x)  # tensor([2.3026, 2.3026])

print("------------------------ torch.log10(tensor, )：以e为底的对数")
x = torch.tensor([10, 10])
print(x)
x = torch.log10(x)
print(x)  # tensor([1., 1.])

print("------------------------ torch.where(condition, input, other)：等价于 java 三元组")
# https://pytorch.org/docs/1.2.0/torch.html#where.log
# Return a tensor of elements selected from either input or other, depending on condition.
x = torch.tensor(10.0)
print(x)
y = torch.tensor(1.0)
print(y)
z = torch.where(torch.tensor(4) > 0.5, x, y)
print(z)

print("------------------------ torch.squeeze(input, dim=None, out=None)：返回的是一个值")
# https://pytorch.org/docs/1.2.0/torch.html#torch.squeeze
# Returns a tensor with all the dimensions of input of size 1 removed.

x = torch.squeeze(z)
print(x)

print("------------------------ torch.gather(input, dim, index, out=None, sparse_grad=False) → Tensor：")
# https://pytorch.org/docs/1.2.0/torch.html#torch.gather


print("------------------------------------------------")
x = torch.zeros(2, 1, 2, 1, 2)
print(x)
print(x.size())
x = torch.zeros(2, 1)
print(x)
print(x.size())

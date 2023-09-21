import torch

# 使用𝐓𝐞𝐧𝐬𝐨𝐫初始化一个𝟏×𝟑的矩阵𝑴和一个𝟐×𝟏的矩阵𝑵，对两矩阵进行减法操作(要求 实现三种不同的形式)，给出结果并分析三种方式的不同(如果出现报错，分析报错的原因)， 同时需要指出在计算过程中发生了什么
x = torch.ones(1, 3)
print(x)
y = torch.rand(2,1)
print(y)
# 减法的三种形式
print(x - y)
print(torch.sub(x, y))
# print(y.sub_(x))


# 利用 𝐓𝐞𝐧𝐬𝐨𝐫创建两个大小分别 𝟑×𝟐 和 𝟒×𝟐的随机数矩阵𝑷和𝑸，要求服从均值为0，标 准差0.01为的正态分布;2 对第二步得到的矩阵 𝑸 进行形状变换得到 𝑸 的转置 𝑸𝑻 ;3 对上述 得到的矩阵 𝑷 和矩阵 𝑸𝑻 求矩阵相乘
# 还可以通过 numpy 库来构建 正态分布
p = torch.normal(mean=0, std=0.01, size=(3, 2))
print(p)
q = torch.normal(mean=0, std=0.01, size=(4, 2))
print(q)

qt = torch.t(q)
print(qt)
mm = torch.mm(p, qt)
print(mm)

# 给定公式 𝑦3 = 𝑦1 + 𝑦2 = 𝑥2 + 𝑥3，且 𝑥 = 1。利用学习所得到的Tensor的相关知识，求𝑦3对𝑥的 梯度





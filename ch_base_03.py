import torch

print('-----------------广播机制')
x = torch.arange(1, 3) # 左闭右开，
print(x)
x = x.view(1,2) # 变成1行2列
print(x)
y = torch.arange(1, 4)
print(y)
y = y.view(3,1) # 变成3行1列
print(y)
print(x+y)


print('-----------------Tensor和NumPy相互转换')
# 我们可以使用numpy()和from_numpy()将Tensor和NumPy中的数组相互转换。但是需要注意的一点 是: 这两个函数所产生的Tensor和NumPy中的数组共享相同的内存(所以他们之间的转换很快)，
# 改变其 中一个时另一个也会改变!
# 还有一个常用的将NumPy中的array转换成Tensor的方法就是torch.tensor(), 需要注意的是，此方法总是 会进行数据拷贝(就会消耗更多的时间和空间)，所以返回的Tensor和原来的数据不再共享内存。
print('-----------------Tensor转NumPy数组')
a = torch.ones(3)
b = a.numpy()
print(a, b)

a += 1
print(a, b)

b += 1
print(a, b)

print('-----------------NumPy数组转Tensor')
import numpy as np

a = np.ones(3)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)

b += 1
print(a, b)
print('使用torch.tensor()将NumPy数组转换成Tensor(不再共享内存)')

c = torch.tensor(a)
a += 1
print(a, c)




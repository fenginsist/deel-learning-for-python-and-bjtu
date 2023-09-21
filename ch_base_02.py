import torch

# Tensor的相关操作

print('-----------------加法形式一')
x = torch.rand(2, 3)
y = torch.rand(2, 3)
print(x+y)

print('-----------------加法形式二')
print(torch.add(x, y))

print('-----------------加法形式三，inplace (原地操作)')
print('-----------------PyTorch操作的inplace版本都有后缀“_” , 例如x.copy_(y), x.t_()')
y.add_(x)
print(y)

print('-----------------索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。')
print(x)
y = x[0,:]
y += 1
print(y)
print(x[0,:]) # 源tensor也被改了

print('-----------------用view()来改变Tensor的形状')
y = x.view(6)
z = x.view(-1,2) # 限定后面是2列，所有就是多行2列。-1所指的维度可以根据其他维度的值推出来，后面是2列，因为是六个数据，所以肯定是三行二列     ??????
print(x)
print(y)
print(z)
print(x.size())
print(y.size())
print(z.size())


# 注意view()返回的新Tensor与源Tensor虽然可能有不同的size，但是是共享data的，也 即更改其中的一个，另外一个也会跟着改变。
# (顾名思义，view仅仅是改变了对这个张量的 观察角度，内部数据并未改变)

print('-----------------view仅仅是改变了对这个张量的 观察角度，内部数据并未改变')
x += 1
print(x)
print(y) # y也加了1

# 如果我们想返回一个真正新的副本(即不共享data内存)该怎么办呢?
# Pytorch还提供了一个reshape()方法可以改变形状，但是此函数并不能保证返回的是其拷贝，所以不推 荐使用。我们推荐先用clone()创造一个副本然后再使用view()。

print('-----------------返回一个真正新的副本(即不共享data内存)：clone（）函数')
print(x)
x_cp = x.clone().view(6)
x -= 1
print(x)
print(x_cp)

print('-----------------对角线元素之和(矩阵的迹)：trace()函数')
x = torch.rand(3,3)
print(x)
print(x.trace())


print('-----------------diag()函数：对角线元素')
print(x)
print(x.diag())

print('-----------------矩阵的上三角/下三角，可指 定偏移量')

print('-----------------torch.mm(x, y)：矩阵乘法，batch的矩阵乘法（参数有点多，需要查看文档）：addmm/addbmm/addmv/addr/baddbmm..')
x = torch.tensor([[1,2], [2,2]])
y = torch.tensor([[2,3], [3,4]])
print(x)
print(y)
print(torch.mm(x, y))
print(torch.mm(torch.mm(x, y), y))
print(torch.addmm(x, y, y))

print('-----------------torch.matmul(x, y)：矩阵乘法，')
# 1、Pytorch矩阵乘法之torch.mul() 、 torch.mm() 及torch.matmul()的区别：https://blog.csdn.net/irober/article/details/113686080
# 2、torch.matmul和torch.mm和*区别：https://blog.csdn.net/Lison_Zhu/article/details/110181622
x = torch.matmul()

print('-----------------转置')
x = torch.rand(3,3)
print(x)
print(torch.t(x))

print('-----------------内积（是数）/外积（矩阵相乘）')
# 理论知识：https://zhuanlan.zhihu.com/p/575908809
#
x = torch.tensor([1,2])
y = torch.tensor([3,3])
print(torch.dot(x,y))

x = torch.tensor([[1,2],[1,2]])
y = torch.tensor([[3,3],[3,3]])
# print(torch.cross(x,y))           #  外积出错

print('-----------------求逆矩阵:报错了')
print(x)
# print(torch.inverse(x))

print('-----------------奇异值分解')
# 待学
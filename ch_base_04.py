import torch

# 自动求梯度



print('-----------------自动求梯度')
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn) # Node ，因为x是直接创建的，所以它没有 grad_fn 没有 grand 函数

y = x + 2
print(y)
print(y.grad_fn)
# <AddBackward0 object at 0x7f7b955ac8b0> y是通过一个加法操作创建的，所以它有一个0为<AddBackward>的grad_fn

z = y * y * 3
out = z.mean()
print(z)
print(z.grad_fn)
print(out)
print(out.grad_fn)

print('-----------------通过.requires_grad_()来用in-place的方式改变requires_grad属性')
a = torch.rand(2, 2)
a = ((a * 3) / (a - 1))
print(a)
print(a.requires_grad) # False,默认就是false
a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn)

print('-----------------梯度，反向传播')
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
z = y * y * 3
print(z)
out = z.mean()
out.backward() # 反向传播，等价于 out.backward(torch.tensor(1.))

print(x.grad) # 输出张量为4.5，我很困惑，为啥呢

print('-----------------梯度，再来反向传播一次，注意grad是累加的')
out2 = x.sum()
out2.backward() # 反向传播,注意grad是累加的,没有清零
print(out2)     # tensor(4., grad_fn=<SumBackward0>)
print(x)        # 都是1
print(x.grad)   # 输出 5.5 为之前的4.5+1，             梯度未清零，累加梯度

print('--------')

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(out3)     # tensor(4., grad_fn=<SumBackward0>)
print(x)        # 都是1
print(x.grad)   # 都是1，因为进行梯度清零了。            梯度清零后，x的梯度为1



print('-----------------y.backward()的具体例子，数据重新开始的')
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2*x
z = y.view(2,2) # 行和列
print(y)
print(z)

print('------')

# 此时 𝒛 不是一个标量，所以在调用backward()时需要传入一个和 𝒛 同形的权重向量 进行加权求和来得到一个标量。
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(v)
print(z)
print(x.grad)
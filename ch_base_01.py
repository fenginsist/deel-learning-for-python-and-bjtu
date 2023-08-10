import torch
print('--------------测试torch 导入是否成功')
x = torch.rand(2,2)
print(x)

print('--------------创建一个2x3的未初始化的Tensor')
x = torch.empty(2, 3)
print(x)

print('--------------创建一个2x3的随机初始化的Tensor')
x = torch.rand(2, 3)
print(x)

print('--------------创建一个2x3的long型全0的Tensor')
# x = torch.zeros(2, 3, dtype=torch.lang)
# print(x)

print('--------------直接根据数据创建')
x = torch.tensor([[5.5, 3],[2.2, 5]])
print(x)

print('----------------------------还可以通过现有的Tensor来创建----------------------------')

print('--------------返回的tensor默认具有相同的torch.dtype和torch.device')
x = x.new_ones(2,3)
print(x)

print('--------------指定新的数据类型')
x = torch.randn_like(x, dtype=torch.float)
print(x)

print('--------------返回的torch.Size其实就是一个tuple, 支持所有tuple的操作。')
print(x.size())
print(x.shape)

print('--------------全1Tensor')
x = torch.ones(2,3)
print(x)

print('--------------对角线为1，其他为0')
x = torch.eye(2,3)
print(x)
import torch

x = torch.tensor(0.0)
print(x)

def penalty_12(w):
    return (w ** 2).sum() / 2


x = torch.rand(10, 1)
print(x)
x = penalty_12(x)
print(x)

true_w= 0.0056 * torch.ones(500,1)
print(true_w.size())
# print(true_w)
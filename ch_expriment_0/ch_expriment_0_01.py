# 实验0 里面通过nn.Module 来实现回归


import torch


class MyLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MyLayer, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数 self.in_features = in_features
        self.outweight_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))  # 由于weights是可以训练的，所以使用Parameter来定义。
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(in_features))  # 由于bias是可以训练 的，所以使用Parameter来定义
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        input_ = torch.pow(input, 2) + self.bias
        y = torch.matmul(input_, self.weight)
        return y


print("end")

# # Tensors
import torch

#
# x = torch.empty(5, 3)
# print(x)
#
# x = torch.zeros(5, 3)
# print(x)
#
# x = torch.rand(5, 3)
# print(x)
#
# x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
# print(x)
#
# x = torch.randn_like(x, dtype=torch.float)  # override dtype!
# print(x)  # result has the same size
#
# print(x.size())  # torch.Size is in fact a tuple, so it supports all tuple operations.
#
# size = list(x.size())
# print(size)
#
# y = torch.rand(5, 3)
# print(y)
# print(x + y)  # same as torch.add x.add(y)
# print(y.add(x))  # will not change y
# print(y)
#
# print(y.add_(x))  # will change y
# print(y)
#
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)
#
# print(x)
# print(x[:, 1])
#
# z = torch.rand(2, 3)
# k = torch.rand(3,2)
# print(z)
# print(z.mm(k))


# 点乘
# x = torch.rand(2, 3)
# y = torch.rand(2, 3)
# print(x)
# print(y)
# print(x.mul(y))
#
# # 矩阵相乘
# x = torch.rand(2, 3)
# y = torch.rand(3, 6)
# print(x)
# print(y)
# print(x.mm(y))
#
# b = x.numpy()
# print(b)
# x.add_(1)
# print(b)
#
# import numpy as np
#
# a = np.ones(5)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)
# print(a)
# print(b)
#
# if torch.cuda.is_available():
#     print('gpu available')
#     device = torch.device('cuda')  # 获取cuda设备
#     y = torch.ones_like(x, device=device)  # 创建tensor声明device
#     x = x.to(device)  # 将tensor发送到设备上
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))

test = torch.rand(7, 7, 32)
print(test.size())
test = test.view(test.size(0), -1)
print(test.size())
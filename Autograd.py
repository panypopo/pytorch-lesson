import torch

# x = torch.ones(2, 2, requires_grad=True)
# print(x)
#
# y = x + 2
# print(y)
#
# # y was created as a result of an operation, so it has a grad_fn.
# print(y.grad_fn)
#
# z = y * y * 3
# out = z.mean()
# print(z, out)
#
# out.backward()  # 因为out是标量 27, 等同out.backward(torch.tensor(1.))
# print(x.grad)

from torch.autograd import Variable

x = Variable(torch.ones(2), requires_grad=True)  # 定义一个变量 vairable是tensor的一个外包装
z = 4 * x * x
y = z.norm() # 求长度 返回输入张量input 的p 范数。
print(y)

y.backward()

print(x.grad) # 返回y关于x的梯度向量
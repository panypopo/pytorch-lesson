import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像通道，6个输出通道，5x5平方卷积
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 最大池(2,2)窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是正方形 可以只指定一个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除批量维度外的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(params)
print(len(params))
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
print(input)
out = net(input)
print(out)

# 清零
net.zero_grad()
# 反向传播， 计算梯度
out.backward(torch.randn(1,10))

output = net(input)
target = torch.rand(10) # 实际值
target = target.view(1,-1)
criterion = nn.MSELoss()

# 损失函数
loss = criterion(output, target)
print(loss)
print(loss.grad_fn)

# 清空所有参数的梯度缓存
net.zero_grad()
print('before:')
print(net.conv1.bias.grad)
# 使用随机梯度进行反向传播计算
loss.backward()
print('after:')
print(net.conv1.bias.grad)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 使用序列工具构建
        # 标准的卷积动作 卷积conv2d -> 归一化 -> 激活 -> 最大池化
        # MNIST数据集的输入为: 28 x 28 x 1 (channel为1，是灰度图)
        # Input的格式： batch x 28 x 28 x 1
        self.conv1 = nn.Sequential(
            # 卷积层 输入1个channel, 16个卷积核, 卷积核大小5X5
            # 加入padding=2，使得输出宽高与输入宽高一致(28 - 5 + 2*2)/1 + 1 = 28
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 归一化 参数为输入的channel
            nn.BatchNorm2d(16),
            # 激活函数
            nn.ReLU(),
            # 池化 2 X 2, 经过池化后 宽高发生了改变
            # 计算 (28-2)/2 + 1 -> 14 x 14 x 16
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            # 第二个卷积层 输入16个channel, 32个卷积核, 卷积核大小5X5
            # 加入padding=2 得输出宽高与输入宽高一致 (14-5+2*2)/1+1 = 14
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            # 归一化
            nn.BatchNorm2d(32),
            # 激活函数
            nn.ReLU(),
            # 池化 2 X 2, 经过池化后 宽高发生了改变
            # 计算 (14-2)/2 + 1 -> 7 x 7 x 32
            nn.MaxPool2d(2)
        )

        # 全连接层 定义有 7*7*32个神经元
        # 把上面的输出全连接，输出成10个分类（MMIST手写字 0-9）
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        # 向前传播算法
        out = self.conv1(x)
        out = self.conv2(out)
        # 上述输出 经过pytorch后的size是 batch_size x channel x width x height
        # 调用view，将输出reshape，变成 batch_size x (channel x width x height) 平铺
        out = out.view(out.size(0), -1)  # reshape ???
        out = self.fc(out)
        return out


class LeNet(nn.Module):
    """
    LeNet: 早期的卷积神经网络,LeNet展示了通过梯度下降训练卷积神经网络可以达到手写数字识别在当时最先进的结果。
    这个奠基性的工作第一次将卷积神经网络推上舞台，为世人所知。
    LeNet分为卷积层块和全连接层块两个部分.
    卷积层块里的基本单位是卷积层后接最大池化层.
    在卷积层块中，每个卷积层都使用 5×5 的窗口，并在输出上使用sigmoid激活函数。
    第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16.
    卷积层块的两个最大池化层的窗口形状均为 2×2 ，且步幅为2。
    全连接层的输入形状将变成二维，其中第一维是小批量中的样本，第二维是每个样本变平后的向量表示，
    且向量长度为通道、高和宽的乘积。全连接层块含3个全连接层。它们的输出个数分别是120、84和10，其中10为输出的类别个数。
    """

    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层
        # 两个卷积单位
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, stride=2)
        )
        # 全连接层，3个Liner
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 16, 120),
            nn.Sigmoid(),
            nn.Linear(120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(84, out_features=10)
        )

    def forward(self, input):
        out = self.conv(input)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out




# 数据相关
num_epochs = 1  # 训练次数
batch_size = 100  # 训练批次
learning_rate = 0.001  # 学习率


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

# 从torchvision.datasets中加载训练和测试数据
train_dataset = datasets.MNIST(
    root="./mnist/",
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

test_dataset = datasets.MNIST(
    root="./mnist/",
    train=False,
    transform=transforms.ToTensor()
)

# 使用DataLoader处理数据 进行Batch训练
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

cnn = LeNet()
# for GPU
if torch.cuda.is_available():
    cnn = cnn.cuda()

# 定义损失函数 和 优化器
# 后续深入
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# 开始训练

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = get_variable(images)
        labels = get_variable(labels)

        outputs = cnn(images)
        loss = loss_func(outputs, labels)
        # 损失函数反向传播计算
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data.item()))

# 测试模型
cnn.eval()  # 测试场景
correct = 0
total = 0

for i, (images, labels) in enumerate(test_loader):
    images = get_variable(images)
    labels = get_variable(labels)
    outputs = cnn(images)
    # 获取最大值以及最大值所在的index index即分类
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()
print(' 测试 准确率: %d %%' % (100 * correct / total))

# 保存模型
torch.save(cnn.state_dict(), 'cnn.pkl')

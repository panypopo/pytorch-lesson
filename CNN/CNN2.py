import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import transforms

"""
AlexNet && VGG && NiN
"""


class AlexNet(nn.Module):
    """
    PS:AlexNet使用的图像高和宽224
    AlexNet使用了8层卷积神经网络，并以很大的优势赢得了ImageNet 2012图像识别挑战赛。
    它首次证明了学习到的特征可以超越手工设计的特征，从而一举打破计算机视觉研究的前状。

    1.AlexNet包含8层变换，其中有5层卷积和2层全连接隐藏层，以及1个全连接输出层。
    AlexNet第一层中的卷积窗口形状是 11×11 。因为ImageNet中绝大多数图像的高和宽均比MNIST图像的高和宽大10倍以上，ImageNet图像的物体占用更多的像素，所以需要更大的卷积窗口来捕获物体。
    第二层中的卷积窗口形状减小到 5×5 ，之后全采用 3×3 。
    此外，第一、第二和第五个卷积层之后都使用了窗口形状为 3×3 、步幅为2的最大池化层。
    而且，AlexNet使用的卷积通道数也大于LeNet中的卷积通道数数十倍。
    紧接着最后一个卷积层的是两个输出个数为4096的全连接层。这两个巨大的全连接层带来将近1 GB的模型参数。

    2.AlexNet将sigmoid激活函数改成了更加简单的ReLU激活函数

    3.AlexNet通过丢弃法来控制全连接层的模型复杂度。

    4.AlexNet引入了大量的图像增广，如翻转、裁剪和颜色变化，从而进一步扩大数据集来缓解过拟合。
    """

    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv = nn.Sequential(
            # kernel_size = 11, (224-11)/4 +1 = 54.25 丢弃后得 54
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            # (54-3)/2 = 26.5 丢弃后得 26
            nn.MaxPool2d(3, stride=2),
            # 众所周知, stride =1, padding = (kernel_size - 1) / 2 的卷积层后宽高不变
            # 这一层得 26 x 26 x 256
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            # (26 -3)/ 2 = 11.5 丢弃后得 11
            # 11 x 11 x 256
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # 11 x 11 x 256
            # (11-3)/2 + 1 = 5
            # 5 x 5 x 256
            nn.MaxPool2d(3, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )

    def forward(self, input):
        out = self.conv(input)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class VGG(nn.Module):
    """
    VGG，它的名字来源于论文作者所在的实验室Visual Geometry Group.
    VGG提出了可以通过重复使用简单的基础块来构建深度模型的思路。

    VGG块:
        VGG块的组成规律是：连续使用数个相同的填充为1、
        窗口形状为 3×3 的卷积层后接上一个步幅为2、窗口形状为 2×2 的最大池化层。
        卷积层保持输入的高和宽不变，而池化层则对其减半。

    VGG计算更慢 更耗显存:我的显存不够一直运行不起来

    """

    def vgg_block(self, num_convs, in_channels, num_channels):
        layers = []
        for _ in range(num_convs):
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1)]
            layers += [nn.ReLU()]
            in_channels = num_channels
        layers += [nn.MaxPool2d(2, 2)]
        blk = nn.Sequential(*layers)
        return blk

    def __init__(self):
        super(VGG, self).__init__()
        # 构造一个VGG网络。它有5个卷积块，前2块使用单卷积层，而后3块使用双卷积层。
        # 第一块的输出通道是64，之后每次对输出通道数翻倍，直到变为512。
        # 因为这个网络使用了8个卷积层和3个全连接层，所以经常被称为VGG-11。
        conv_args = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
        layers = []
        for (num_convs, in_channels, num_channels) in conv_args:
            layers += [self.vgg_block(num_convs, in_channels, num_channels)]
        self.conv = nn.Sequential(*layers)

        # 上面输出
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )

    def forward(self, input):
        out = self.conv(input)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class NiN(nn.Module):
    """
    NiN: 它提出了另外一个思路，即串联多个由卷积层和“全连接”层构成的小网络来构建一个深层网络.
    NiN块:
        我们知道，卷积层的输入和输出通常是四维数组（样本，通道，高，宽），
        而全连接层的输入和输出则通常是二维数组（样本，特征）。
        如果想在全连接层后再接上卷积层，则需要将全连接层的输出变换为四维。

        1×1 卷积层。它可以看成全连接层，其中空间维度（高和宽）上的每个元素相当于样本，通道相当于特征。
        因此，NiN使用 1×1 卷积层来替代全连接层，从而使空间信息能够自然传递到后面的层中去。

        卷积层-> 1x1卷积层->卷积层-> 1x1卷积层
        NiN块是NiN中的基础块。它由一个卷积层加两个充当全连接层的 1×1 卷积层串联而成。
        卷积层-> 1x1卷积层-> 1x1卷积层

    NiN模型:
        NiN与AlexNet一样使用5层卷积
        NiN使用卷积窗口形状分别为 11×11 、 5×5 和 3×3 的卷积层，相应的输出通道数也与AlexNet中的一致。
        每个NiN块后接一个步幅为2、窗口形状为 3×3 的最大池化层。
        NiN去掉了AlexNet最后的3个全连接层，取而代之地，
        NiN使用了输出通道数等于标签类别数的NiN块，然后使用全局平均池化层对每个通道中所有元素求平均并直接用于分类。
        这里的全局平均池化层即窗口形状等于输入空间维形状的平均池化层。
    """

    def nin_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        blk = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU()
        )
        return blk

    def __init__(self):
        super(NiN, self).__init__()
        self.conv = nn.Sequential(
            self.nin_block(in_channels=1, out_channels=96, kernel_size=11, stride=4),
            nn.MaxPool2d(3, 2),
            self.nin_block(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.MaxPool2d(3, 2),
            self.nin_block(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.MaxPool2d(3, 2),
            self.nin_block(in_channels=384, out_channels=10, kernel_size=3, padding=1),
            nn.AvgPool2d(5)
        )

    def forward(self, input):
        out = self.conv(input)
        out = out.view(out.size(0), 10)
        return out


# 数据相关
num_epochs = 1  # 训练次数
batch_size = 5  # 训练批次
learning_rate = 0.001  # 学习率


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


train_dataset = datasets.FashionMNIST(
    root="./fashionmnist/",
    train=True,
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
    download=False
)

test_dataset = datasets.FashionMNIST(
    root="./fashionmnist/",
    train=False,
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
    download=False
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

cnn = VGG()
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
torch.save(cnn.state_dict(), 'cnn2.pkl')

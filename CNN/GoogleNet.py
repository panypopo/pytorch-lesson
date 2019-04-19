import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import transforms

"""
GoogleNet
"""


class Inception(nn.Module):
    """
    Inception块:GoogLeNet中的基础卷积块叫作Inception块
        输入
        1x1卷积层   1x1卷积层    1x1卷积层    3X3最大池化
                   3x3卷积层    5x5卷积层     1x1卷积层
        通道合并层

        Inception块里有4条并行的线路。前3条线路使用窗口大小分别是 1×1 、 3×3 和 5×5 的卷积层来抽取不同空间尺寸下的信息，
        其中中间2个线路会对输入先做 1×1 卷积来减少输入通道数，以降低模型复杂度。
        第四条线路则使用 3×3 最大池化层，后接 1×1 卷积层来改变通道数。

    """

    def __init__(self, channels1, channels2, channels3, channels4):
        super(Inception, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=channels1[0], out_channels=channels1[1], kernel_size=1),
            nn.ReLU()
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=channels2[0], out_channels=channels2[1], kernel_size=1),
            nn.ReLU(),
            # 为保持输出的尺寸大小一致，这里要加padding
            nn.Conv2d(in_channels=channels2[1], out_channels=channels2[2], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=channels3[0], out_channels=channels3[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels3[1], out_channels=channels3[2], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels4[0], out_channels=channels4[1], kernel_size=1),
            nn.ReLU()
        )

    def forward(self, input):
        out1 = self.c1(input)
        out2 = self.c2(input)
        out3 = self.c3(input)
        out4 = self.c4(input)
        out = torch.cat([out1, out2, out3, out4], 1)  # 在通道维上连结输出
        return out


class GoogleNet(nn.Module):
    """
        GoogLeNet吸收了NiN中网络串联网络的思想，并在此基础上做了很大改进。
        Inception块:GoogLeNet中的基础卷积块叫作Inception块

        GoogLeNet模型:
        GoogLeNet跟VGG一样，在主体卷积部分中使用5个模块（block），
        每个模块之间使用步幅为2的 3×3 最大池化层来减小输出高宽。

    """

    def __init__(self):
        super(GoogleNet, self).__init__()
        # 第一模块使用一个64通道的 7×7 卷积层
        bk1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # 64 * 56 * 56
        )

        # 第二模块使用2个卷积层：首先是64通道的 1×1 卷积层，然后是将通道增大3倍的 3×3 卷积层。它对应Inception块中的第二条线路。
        bk2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            # 192 * 56 * 56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # 192 * 28 * 28
        )

        # 第三模块串联2个完整的Inception块。第一个Inception块的输出通道数为 64+128+32+32=256 ，
        # 其中4条线路的输出通道数比例为 64:128:32:32=2:4:1:1 。其中第二、第三条线路先分别将输入通道数减小至 96/192=1/2 和 16/192=1/12 后，
        # 再接上第二层卷积层。第二个Inception块输出通道数增至 128+192+96+64=480 ，每条线路的输出通道数之比为 128:192:96:64=4:6:3:2 。
        # 其中第二、第三条线路先分别将输入通道数减小至 128/256=1/2 和 32/256=1/8 。
        bk3 = nn.Sequential(
            Inception((192, 64), (192, 96, 128), (192, 16, 32), (192, 32)),
            Inception((256, 128), (256, 128, 192), (256, 64, 96), (256, 64)),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第四模块更加复杂。它串联了5个Inception块，其输出通道数分别是 192+208+48+64=512 、 160+224+64+64=512 、 128+256+64+64=512 、
        # 112+288+64+64=528 和 256+320+128+128=832 。
        # 这些线路的通道数分配和第三模块中的类似，首先含 3×3 卷积层的第二条线路输出最多通道，其次是仅含 1×1 卷积层的第一条线路，
        # 之后是含 5×5 卷积层的第三条线路和含 3×3 最大池化层的第四条线路。其中第二、第三条线路都会先按比例减小通道数。
        # 这些比例在各个Inception块中都略有不同。
        bk4 = nn.Sequential(
            Inception((480, 192), (480, 96, 208), (480, 16, 48), (480, 64)),
            Inception((512, 160), (512, 112, 224), (512, 24, 64), (512, 64)),
            Inception((512, 128), (512, 128, 256), (512, 24, 64), (512, 64)),
            Inception((512, 112), (512, 144, 288), (512, 32, 64), (512, 64)),
            Inception((528, 256), (528, 160, 320), (528, 32, 128), (528, 128)),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第五模块有输出通道数为 256+320+128+128=832 和 384+384+128+128=1024 的两个Inception块。
        # 其中每条线路的通道数的分配思路和第三、第四模块中的一致，只是在具体数值上有所不同。需要注意的是，第五模块的后面紧跟输出层，
        # 该模块同NiN一样使用全局平均池化层来将每个通道的高和宽变成1。最后我们将输出变成二维数组后接上一个输出个数为标签类别数的全连接层。
        bk5 = nn.Sequential(
            Inception((832, 256), (832, 160, 320), (832, 32, 128), (832, 128)),
            Inception((832, 384), (832, 192, 384), (832, 48, 128), (832, 128)),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv = nn.Sequential(
            bk1, bk2, bk3, bk4, bk5
        )

        def fc(num_of_features):
            # print('num_of_features:', num_of_features)
            return nn.Linear(num_of_features, 10)

        self.fc_func = fc

    def forward(self, input):
        out = self.conv(input)
        fc = self.fc_func(self.num_flat_features(out))
        if torch.cuda.is_available():
            fc = fc.cuda()
        out = out.view(out.size(0), -1)
        out = fc(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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

cnn = GoogleNet()
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
torch.save(cnn.state_dict(), 'googleNet.pkl')

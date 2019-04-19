import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import transforms

"""
    残差网络:
    在实践中，神经网络添加过多的层后训练误差往往不降反升。
    即使利用批量归一化带来的数值稳定性使训练深层模型更加容易，该问题仍然存在。
    针对这一问题，何恺明等人提出了残差网络（ResNet）
    14年的VGG才19层，而15年的ResNet多达152层！

    深度网络的退化问题：网络深度增加时，网络准确度出现饱和，甚至出现下降

    残差学习：
    有一个浅层网络，你想通过向上堆积新层来建立深层网络，
    一个极端情况是这些增加的层什么也不学习，仅仅复制浅层网络的特征，即这样新层是恒等映射（Identity mapping）
    对于一个堆积层结构， 输入x时其学习到的特征记为 H(x)。
    我们希望其可以学习到残差 F(x)=H(x)-x ，这样其实原始的学习特征是 H(x) = F(x)+x。
    残差学习相比原始特征直接学习更容易，为什么残差学习相对更容易？从直观上看残差学习需要学习的内容少，因为残差一般会比较小，学习难度小点。


"""


class Residual(nn.Module):
    """
    残差块:
    残差块里首先有2个有相同输出通道数的 3×3 卷积层。
    每个卷积层后接一个批量归一化层和ReLU激活函数。
    然后我们将输入跳过这两个卷积运算后直接加在最后的ReLU激活函数前。
    这样的设计要求两个卷积层的输出与输入形状一样，从而可以相加。
    如果想改变通道数，就需要引入一个额外的 1×1 卷积层来将输入变换成需要的形状后再做相加运算。

    """

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(Residual, self).__init__(**kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        self.al = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        if self.conv3:
            input = self.conv3(input)
        out = self.al(out + input)
        return out


class ResNet(nn.Module):
    """
    残差网络：

    """

    def __init__(self):
        super(ResNet, self).__init__()
        # ResNet的前两层在输出通道数为64、步幅为2的 7×7 卷积层后接步幅为2的 3×3 的最大池化层。
        # 每个卷积层后增加的批量归一化层。
        bk1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, stride=2, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet则使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。
        # 第一个模块的通道数同输入通道数一致。由于之前已经使用了步幅为2的最大池化层，所以无须减小高和宽。
        # 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。
        # 这里每个模块使用两个残差块。
        # ResNet-18

        r1 = nn.Sequential(
            Residual(in_channels=64, out_channels=64, stride=1),
            Residual(in_channels=64, out_channels=64, stride=1)
        )

        r2 = nn.Sequential(
            Residual(in_channels=64, out_channels=128, stride=2),
            Residual(in_channels=128, out_channels=128, stride=1)
        )

        r3 = nn.Sequential(
            Residual(in_channels=128, out_channels=256, stride=2),
            Residual(in_channels=256, out_channels=256, stride=1)
        )

        r4 = nn.Sequential(
            Residual(in_channels=256, out_channels=512, stride=2),
            Residual(in_channels=512, out_channels=512, stride=1)
        )
        # 加入全局平均池化层后接上全连接层输出
        self.conv = nn.Sequential(
            bk1, r1, r2, r3, r4, nn.AvgPool2d(kernel_size=1)
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

cnn = ResNet()
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
torch.save(cnn.state_dict(), 'ResNet.pkl')

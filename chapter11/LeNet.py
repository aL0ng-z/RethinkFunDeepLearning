import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # C1: 输入1通道，输出6通道，卷积核5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # S2: 平均池化层
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 输入6通道，输出16通道
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # S4: 平均池化
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C5: 全连接等价层（输入16×5×5 -> 输出120）
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        # F6: 全连接层
        self.fc1 = nn.Linear(120, 84)
        # Output: 输出10类
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))     # C1 + 激活
        x = self.pool1(x)             # S2
        x = F.tanh(self.conv2(x))     # C3 + 激活
        x = self.pool2(x)             # S4
        x = F.tanh(self.conv3(x))     # C5 + 激活
        x = x.view(-1, 120)           # 展平
        x = F.tanh(self.fc1(x))       # F6
        x = self.fc2(x)               # 输出层
        return x
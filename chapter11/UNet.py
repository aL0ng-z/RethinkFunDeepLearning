import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """两个连续的 3x3 Conv（padding=1）+ BN + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 编码路径
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = DoubleConv(512, 1024)  # 最底层

        # 解码路径
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 上采样 ×2
        self.conv6 = DoubleConv(1024, 512)  # 拼接后通道数变为 512+512=1024

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # 编码
        c1 = self.conv1(x)      # c1: [B, 64, H, W]
        p1 = self.pool1(c1)     # p1: [B, 64, H/2, W/2]
        c2 = self.conv2(p1)     # c2: [B,128, H/2,W/2]
        p2 = self.pool2(c2)     # p2: [B,128, H/4,W/4]
        c3 = self.conv3(p2)     # c3: [B,256, H/4,W/4]
        p3 = self.pool3(c3)     # p3: [B,256, H/8,W/8]
        c4 = self.conv4(p3)     # c4: [B,512, H/8,W/8]
        p4 = self.pool4(c4)     # p4: [B,512, H/16,W/16]
        c5 = self.conv5(p4)     # c5: [B,1024,H/16,W/16]

        # 解码，第 1 级上采样
        u6 = self.up6(c5)       # u6: [B,512, H/8,W/8]
        # 直接拼接 c4（[B,512,H/8,W/8]）和 u6
        u6 = torch.cat([c4, u6], dim=1)  # 拼接后：[B,1024,H/8,W/8]
        c6 = self.conv6(u6)     # c6: [B,512, H/8,W/8]

        # 解码，第 2 级
        u7 = self.up7(c6)       # u7: [B,256, H/4,W/4]
        u7 = torch.cat([c3, u7], dim=1)  # [B,512, H/4,W/4]
        c7 = self.conv7(u7)     # [B,256, H/4,W/4]

        # 解码，第 3 级
        u8 = self.up8(c7)       # [B,128, H/2,W/2]
        u8 = torch.cat([c2, u8], dim=1)  # [B,256, H/2,W/2]
        c8 = self.conv8(u8)     # [B,128, H/2,W/2]

        # 解码，第 4 级
        u9 = self.up9(c8)       # [B,64, H, W]
        u9 = torch.cat([c1, u9], dim=1)  # [B,128,H,W]
        c9 = self.conv9(u9)     # [B,64, H, W]

        # 最后一层 1x1 卷积，得到最终预测
        out = self.final_conv(c9)  # [B,out_ch,H,W]
        return out
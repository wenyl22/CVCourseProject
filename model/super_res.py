import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperResModel(nn.Module):
    def __init__(self):
        super(SuperResModel, self).__init__()
        # 编码器
        self.enc1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # 中间层
        self.mid = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 解码器
        self.dec1 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        # 输出层
        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 编码器
        enc1 = self.relu(self.enc1(x))
        enc2 = self.relu(self.enc2(enc1))
        # 中间层
        mid = self.relu(self.mid(enc2))
        # 解码器
        dec1 = F.interpolate(mid, scale_factor=2, mode='nearest')
        dec1 = self.relu(self.dec1(dec1))
        # 跳跃连接
        dec1 = dec1 + enc1
        dec2 = self.dec2(dec1)
        return dec2
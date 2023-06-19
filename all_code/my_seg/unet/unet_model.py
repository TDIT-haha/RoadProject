""" Full assembly of the parts to form the complete network """
import torch

from unet.unet_parts import *
from unet.My_block import MyBlock

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 64)
        self.myblock1 = MyBlock(64,64)
        self.down2 = Down(64, 128)
        self.myblock2 = MyBlock(64,128)
        self.down3 = Down(128, 256)
        self.myblock3 = MyBlock(128,256)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2b = self.myblock1(x1)
        nx2 = torch.cat([x2b, x2], dim=1)
        x3 = self.down2(x2)
        x3b = self.myblock2(x2b)
        nx3 = torch.cat([x3b, x3], dim=1)
        x4 = self.down3(x3)
        x4b = self.myblock3(x3b)
        nx4 = torch.cat([x4, x4], dim=1)
        x5 = self.down4(nx4)

        x = self.up1(x5, nx4)
        x = self.up2(x, nx3)
        x = self.up3(x, nx2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    model = UNet(3,5)
    input = torch.randn(2,3,540,960)
    output = model(input)
    print(output.shape)
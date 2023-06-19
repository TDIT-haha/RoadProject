import torch
import torch.nn as nn
import torch.nn.functional as F
from unet.eca_module import eca_layer

def get_activation(name="silu", inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == "silu":
            module = nn.SiLU(inplace=inplace)
        elif name == "relu":
            module = nn.ReLU(inplace=inplace)
        elif name == "lrelu":
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'swish':
            module = Swish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        else:
            raise AttributeError("Unsupported act type: {}".format(name))
        return module
    elif isinstance(name, nn.Module):
        return name
    else:
        raise AttributeError("Unsupported act type: {}".format(name))

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)

class ConvBNLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=1,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(ch_out, )
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MyBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super(MyBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=2, padding=1, act=None)
        self.conv2 = ConvBNLayer(ch_in, ch_out, 1, stride=2, padding=0, act=None)
        self.eca = eca_layer(ch_out*2, 3)
        self.act = get_activation(act)

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        y = self.eca(y)
        y = self.act(y)
        return y

if __name__ == '__main__':
    model = MyBlock(ch_in=3,ch_out=256,act='relu')
    input = torch.randn(2,3,512,512)
    output = model(input)
    print(output.shape)
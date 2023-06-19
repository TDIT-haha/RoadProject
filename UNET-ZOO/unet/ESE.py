import torch
import torch.nn as nn
from unet.My_block import get_activation

class ESE(nn.Module):
    def __init__(self, channels, act='hardsigmoid'):
        super(ESE, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1,padding=0)
        self.act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

if __name__ == '__main__':
    model = ESE(3)
    input = torch.randn(2,3,512,512)
    output = model(input)
    print(output.shape)
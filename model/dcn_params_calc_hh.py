import torch
import torch.nn as nn

from DCNv2.dcn_v2 import DCN_sep
from thop import profile


class DCN_Params_Calc(nn.Module):
    def __init__(self, in_channels, scale=384, groups=8):
        super().__init__()
        self.dcn = DCN_sep(in_channels, in_channels, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.cnn = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, dilation=1)

    def forward(self, x1, x2=0):
        return self.cnn(x1)


if __name__ == '__main__':
    model = DCN_Params_Calc(in_channels=256).cuda()

    x1 = torch.randn((1, 256, 64, 64)).cuda()
    x2 = torch.randn((1, 256, 64, 64)).cuda()
    flops, params = profile(model, inputs=(x1,))
    print(' Number of parameters:%.4f M' % (params / 1e6))
    print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))
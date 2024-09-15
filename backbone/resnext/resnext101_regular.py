import torch
from torch import nn

from backbone.resnext import resnext_101_32x4d_
from collections import OrderedDict


class ResNeXt101(nn.Module):
    def __init__(self, backbone_path):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d_.resnext_101_32x4d
        if backbone_path is not None:
            # weights = torch.load(backbone_path)
            # net.load_state_dict(weights, strict=True)

            state_dict = torch.load(backbone_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                temp = k[0:9]
                if temp.__eq__('features.'):
                    name = k[9:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
                    new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
                else:
                    name = k[12:]
                    name = '10.1.' + name
                    new_state_dict[name] = v
            net.load_state_dict(new_state_dict, strict=True)

            print("Load ResNeXt Weights Succeed!")

        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4

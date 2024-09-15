import torch
import torch.nn.functional as F
from torch import nn

from bottleneck_transformer_pytorch import BottleStack

from backbone.resnext.resnext101_regular import ResNeXt101
from config import *

from model.Module.GGNBlock import *


###################################################################
# ########################## NETWORK ##############################
###################################################################
class GGN(nn.Module):
    def __init__(self, backbone_path=None):
        super(GGN, self).__init__()
        resnext = ResNeXt101(backbone_path)  # 神经网络模型backbone
        self.layer0 = resnext.layer0  # 0-3
        self.layer1 = resnext.layer1  # 3-5
        self.layer2 = resnext.layer2  # 5
        self.layer3 = resnext.layer3  # 6
        self.layer4 = resnext.layer4  # 7


        # 4个反卷积 升维参数怪怪的 ans:inchannel outchannel 和 图像的尺寸没有啥关系
        self.up_4 = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 4, 2, 1), nn.BatchNorm2d(1024), nn.ReLU())
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.up_2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.up_1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())

        # merge
        self.layer4_predict = nn.Conv2d(2048, 1, 3, 1, 1)
        self.layer3_predict = nn.Conv2d(1024, 1, 3, 1, 1)
        self.layer2_predict = nn.Conv2d(512, 1, 3, 1, 1)
        self.layer1_predict = nn.Conv2d(256, 1, 3, 1, 1)

        # inplace  True 不创建新的对象，对原始对象进行修改
        # False
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4_f = layer4
        up_4 = self.up_4(layer4_f)
        layer3_high = F.upsample(up_4, size=[scale//16, scale//16], mode='bilinear', align_corners=True)
        layer3_f = layer3 + layer3_high

        up_3 = self.up_3(final_contrast_3)
        layer2_high = F.upsample(up_3, size=[scale // 8, scale // 8], mode='bilinear', align_corners=True)
        layer2_f = layer2 + layer2_high

        up_2 = self.up_2(final_contrast_2)
        layer1_high = F.upsample(up_2, size=[scale // 4, scale // 4], mode='bilinear', align_corners=True)
        layer1_f = layer1 + layer1_high

        up_1 = self.up_1(final_contrast_1)

        # 1-4 gst
        layer4_gst_predict = self.layer4_gst_predict(up_4)
        layer3_gst_predict = self.layer3_gst_predict(up_3)
        layer2_gst_predict = self.layer2_gst_predict(up_2)
        layer1_gst_predict = self.layer1_gst_predict(up_1)
        layer4_gst_predict = F.upsample(layer4_gst_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_gst_predict = F.upsample(layer3_gst_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_gst_predict = F.upsample(layer2_gst_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_gst_predict = F.upsample(layer1_gst_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        final_gst_features = torch.cat((x, layer4_gst_predict, layer3_gst_predict, layer2_gst_predict, layer1_gst_predict), 1)
        final_gst_predict = self.gst_refinement(final_gst_features)  # 一个卷积


        final_r2_mask_predict = self.m2_refinement(final_r2_mask_features)  # 一个卷积


        # 1-4 merge
        layer4_merge_predict = self.layer4_predict(final_contrast_4)
        layer3_merge_predict = self.layer3_predict(final_contrast_3)
        layer2_merge_predict = self.layer2_predict(final_contrast_2)
        layer1_merge_predict = self.layer1_predict(final_contrast_1)
        layer4_predict = F.upsample(layer4_merge_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.upsample(layer3_merge_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.upsample(layer2_merge_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.upsample(layer1_merge_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        final_features = torch.cat((x, layer4_predict, layer3_predict, layer2_predict, layer1_predict), 1)
        final_predict = self.merge_refinement(final_features)  # 一个卷积



        # if self.training:
        # return layer4_predict, layer3_predict, layer2_predict, layer1_predict, layer0_edge, final_predict
        # return final_predict

        return F.sigmoid(layer4_predict), F.sigmoid(layer3_predict), F.sigmoid(layer2_predict), F.sigmoid(layer1_predict), F.sigmoid(final_predict)



        # 返回 f4-f1 的预测  edge的预测  final的预测



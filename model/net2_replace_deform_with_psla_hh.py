import torch.nn.functional as F
from DCNv2.dcn_v2 import *
from collections import OrderedDict
from model_sem import SEM_Net
from torch.autograd import Variable
import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image
from backbone.swin_transformer.swin_transformer import SwinTransformer_demo
import numpy as np
# 用于特征图可视化
from visualizer import get_local
from thop import profile
import torch.nn.functional as F

"""
PAMI重投后，应审稿人和港城大意见，将可变性卷积替换成PSLA结构，此方法本质上同non-local存在同样的理论缺陷。
PSLA部分是由港城大实现(更像GPT-4实现)，此处将获取“米”字型采样区域的方法改为了静态方法，以提高效率。

Author: huang hao
Date: 2024/4/1
"""


class ProgressiveSparseLocalAttention(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()

        # The embedding function f(·) and g(·) in Equ. (1) are implemented with 1 × 1 convolution layers with 256 filters
        self.embedding_f = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1)
        self.embedding_g = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1)

        # 初始化embedding
        nn.init.xavier_uniform_(self.embedding_f.weight)
        if self.embedding_f.bias is not None:
            nn.init.constant_(self.embedding_f.bias, 0.0)
        nn.init.xavier_uniform_(self.embedding_g.weight)
        if self.embedding_g.bias is not None:
            nn.init.constant_(self.embedding_g.bias, 0.0)

    def forward(self, Ft, Ft_epsilon):
        """
        输入：
        - Ft, Ft_epsilon: 两张特征图，形状为 (batch_size, in_channels, height, width)

        输出：
        - output_features: 经过Progressive Sparse Local Attention处理后的特征图，形状为 (batch_size, 256, height, width)
        """

        Embedding_t_epsilon = self.embedding_g(Ft_epsilon)
        Embedding_t = self.embedding_f(Ft)

        # 初始化对齐的特征图
        batch_size, channels, height, width = Ft_epsilon.shape
        aligned_feature = torch.zeros_like(Ft_epsilon)

        for x1 in range(height):
            for y1 in range(width):
                # Step 1: 找到Ft中的指定区域Φ
                phi_region = self._find_phi_region(x1, y1, width, height, d=4)
                Ft_phi = [Ft[:, :, p[0], p[1]] for p in phi_region]

                # Step 2: 计算p1和Φ中每个单元p2的特征匹配度, Softmax归一化得到权重
                feature_affinities_list = []
                for i, (x2, y2) in enumerate(phi_region):
                    # 计算g(Ft_epsilon(x1, y1))和f(Ft(x2, y2))的内积
                    embedding_t_epsilon_x1y1 = Embedding_t_epsilon[:, :, x1, y1]  # [b,c]
                    embedding_t_x2y2 = Embedding_t[:, :, x2, y2]  # [b,c]
                    affinity = torch.sum(embedding_t_epsilon_x1y1 * embedding_t_x2y2, dim=1)
                    feature_affinities_list.append(affinity)

                # 应用 Softmax 归一化得到权重
                feature_affinities = torch.stack(feature_affinities_list)
                weights = F.softmax(feature_affinities, dim=0)

                # Step 3:加权求和得到对齐的特征
                aligned_feature[:, :, x1, y1] = torch.sum(weights.unsqueeze(-1) * torch.stack(Ft_phi, dim=0), dim=0)

        return aligned_feature

    @staticmethod
    def _find_phi_region(x1, y1, width, height, d=4):
        phi_region = []
        phi_region.append((x1, y1))
        for s in range(1, d + 1):
            for a in [-s, 0, s]:
                for b in [-s, 0, s]:
                    if a != 0 or b != 0:
                        x = x1 + a
                        y = y1 + b
                        if 0 <= x < width and 0 <= y < height:
                            phi_region.append((x, y))
        return phi_region


def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
    if useBN:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU()
            # nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU()
        )


def upsample(ch_coarse, ch_fine, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        # nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ConvTranspose2d(ch_coarse, ch_fine, kernel_size, stride, padding, bias=False),
        nn.ReLU()
    )


class GDM(nn.Module):
    def __init__(self, in_channels, scale=384, groups=8):
        super(GDM, self).__init__()

        self.in_channels = in_channels
        self.scale = scale

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.plsa = ProgressiveSparseLocalAttention(in_channels)

        self.pred1 = nn.Conv2d(self.in_channels, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.pred2 = nn.Conv2d(self.in_channels, 1, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        r1_feature = self.conv1(x)
        r2_feature = self.conv2(x)
        # transforms.ToPILImage()(F.sigmoid(torch.chunk(r1_feature.squeeze(0), 128, dim=0)[127])).show()

        r2_feature_hat = self.plsa(r1_feature, r2_feature)
        per = r2_feature_hat - r2_feature

        r1_mask = self.pred1(r1_feature)
        r2_mask = self.pred2(r2_feature_hat + r2_feature)
        merge = torch.cat((r1_feature, r2_feature), dim=1)

        r1_mask = F.upsample(r1_mask, size=[self.scale, self.scale], mode='bilinear', align_corners=True)
        r2_mask = F.upsample(r2_mask, size=[self.scale, self.scale], mode='bilinear', align_corners=True)
        r1_mask = F.sigmoid(r1_mask)
        r2_mask = F.sigmoid(r2_mask)

        return merge, r1_mask, r2_mask, per


class SEM(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(SEM, self).__init__()
        # ---------------- Encoder ----------------------
        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(inc_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # ---------------- Decoder ----------------------
        self.deconv4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.deconv0 = nn.Conv2d(64, 2, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, r1, r2):
        # ---------------- Encoder ----------------------
        hx = torch.cat((r1, r2), dim=1)
        hx = self.conv0(hx)

        hx1 = self.conv1(hx)
        hx = self.pool1(hx1)

        hx2 = self.conv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.conv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.conv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.conv5(hx)

        # ---------------- Decoder ----------------------
        hx = self.upsample(hx5)

        d4 = self.deconv4(torch.cat((hx, hx4), 1))
        hx = self.upsample(d4)

        d3 = self.deconv3(torch.cat((hx, hx3), 1))
        hx = self.upsample(d3)

        d2 = self.deconv2(torch.cat((hx, hx2), 1))
        hx = self.upsample(d2)

        d1 = self.deconv1(torch.cat((hx, hx1), 1))
        output = self.deconv0(d1)

        # output = self.fuse(torch.cat((ref, mask), 1))

        # h, w = torch.split(output, [1, 1], 1)
        return output


class UpSampleBlock(nn.Module):
    """
    为生成gedfusion所设计的模块，在该网络可删除
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, [384, 384], mode='bilinear', align_corners=True)


class Net(nn.Module):
    def __init__(self,
                 backbone_path=None,
                 scale=384,
                 img_names=None):
        super(Net, self).__init__()
        self.scale = scale
        self.img_names = img_names
        self.count = 0

        # the net for ghost cues
        GED_swin = SwinTransformer_demo(img_size=384, embed_dim=128, depths=[2, 2, 16, 2],
                                        num_heads=[4, 8, 16, 32], window_size=12)

        if backbone_path is not None:
            state_dict = torch.load(backbone_path)
            pretrained_dict = state_dict['model']
            print("---start load pretrained modle of swin encoder---")
            GED_swin.load_state_dict(pretrained_dict, strict=False)

        self.ged_pebed = GED_swin.patch_embed
        self.ged_pos_drop = GED_swin.pos_drop
        self.ged_layer0 = GED_swin.layers[0]
        self.ged_layer1 = GED_swin.layers[1]
        self.ged_layer2 = GED_swin.layers[2]
        self.ged_layer3 = GED_swin.layers[3]

        self.GDM3 = GDM(1024)
        self.GDM2 = GDM(512)
        self.GDM1 = GDM(256)
        self.GDM0 = GDM(128)

#        SEM_Net_demo = SEM_Net()

#        if backbone_path is not None:
#            state_dict = torch.load('/data/Lee/VDR1.1/DCNv2/logs/Ver0.4.2.s1/train_min.pth')
#            new_state_dict = OrderedDict()
#            for k, v in state_dict.items():
#                name = k[0:3]
#                if name == 'SEM':
#                    new_state_dict[k] = v
#            print("---start load pretrained modle of swin encoder---")
#            SEM_Net_demo.load_state_dict(new_state_dict, strict=False)

        self.SEM3 = SEM(2, 64)
        self.SEM2 = SEM(2, 64)
        self.SEM1 = SEM(2, 64)
        self.SEM0 = SEM(2, 64)

        self.ged_down3 = nn.Conv2d(1024 * 2 + 2, 1024, 1, 1, 0)
        self.ged_down2 = nn.Conv2d(512 * 2 + 2, 512, 1, 1, 0)
        self.ged_down1 = nn.Conv2d(256 * 2 + 2, 256, 1, 1, 0)
        self.ged_down0 = nn.Conv2d(128 * 2 + 2, 128, 1, 1, 0)

        self.ged_up_32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.ged_up_21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.ged_up_10 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.ged_pred0 = nn.Conv2d(128, 1, 3, 1, 1)
        self.ged_pred1 = nn.Conv2d(256, 1, 3, 1, 1)
        self.ged_pred2 = nn.Conv2d(512, 1, 3, 1, 1)
        self.ged_pred3 = nn.Conv2d(1024, 1, 3, 1, 1)
        self.ghost_final_pred = nn.Conv2d(3 + 4, 1, 3, 1, 1)
        self.r1_pred = nn.Conv2d(4, 1, 3, 1, 1)
        self.r2_pred = nn.Conv2d(4, 1, 3, 1, 1)

        # subnet for glass segmentation
        GSD_swin = SwinTransformer_demo(img_size=384, embed_dim=128, depths=[2, 2, 16, 2],
                                        num_heads=[4, 8, 16, 32], window_size=12)

        if backbone_path is not None:
            state_dict = torch.load(backbone_path)
            pretrained_dict = state_dict['model']
            GSD_swin.load_state_dict(pretrained_dict, strict=False)
            weight1 = GSD_swin.patch_embed.proj.weight.clone()

            new_first_layer = torch.nn.Conv2d(4, 128, 4, 4)
            # print(new_first_layer.weight.size())
            new_first_layer.weight[:,:3,:,:].data[...] = Variable(weight1, requires_grad=True)
            GSD_swin.patch_embed.proj = new_first_layer

        self.pebed = GSD_swin.patch_embed
        self.pos_drop = GSD_swin.pos_drop
        self.layer0 = GSD_swin.layers[0]
        self.layer1 = GSD_swin.layers[1]
        self.layer2 = GSD_swin.layers[2]
        self.layer3 = GSD_swin.layers[3]

        self.up_32 = upsample(1024, 512)
        self.up_21 = upsample(512, 256)
        self.up_10 = upsample(256, 128)
        self.up_final = upsample(128, 64, kernel_size=8, stride=4, padding=2)

        self.conv2m = add_conv_stage(1024, 512, useBN=True)
        self.conv1m = add_conv_stage(512, 256, useBN=True)
        self.conv0m = add_conv_stage(256, 128, useBN=True)

        self.final_pred = nn.Conv2d(64, 1, 3, 1, 1)
        self.pred0 = nn.Conv2d(128, 1, 3, 1, 1)
        self.pred1 = nn.Conv2d(256, 1, 3, 1, 1)
        self.pred2 = nn.Conv2d(512, 1, 3, 1, 1)
        self.pred3 = nn.Conv2d(1024, 1, 3, 1, 1)

        # 为生成gedfusion，在该Net中可删除
        self.ged_swin_encode_up4 = UpSampleBlock(1024)
        self.ged_swin_encode_up3 = UpSampleBlock(512)
        self.ged_swin_encode_up2 = UpSampleBlock(256)
        self.ged_swin_encode_up1 = UpSampleBlock(128)

    @get_local('swinOut')
    def forward(self, x):
        # GED
        input = x
        b, c, h, w = x.shape
        x = self.ged_pebed(x)
        x = self.ged_pos_drop(x)
        # torchvision.transforms.ToPILImage()(torch.nn.functional.sigmoid(torch.chunk(ged_layer0.squeeze(0), 144, dim=0)[143])).show()
        ged_layer0, ged_layer0_d = self.ged_layer0(x)
        ged_layer1, ged_layer1_d = self.ged_layer1(ged_layer0_d)
        ged_layer2, ged_layer2_d = self.ged_layer2(ged_layer1_d)
        ged_layer3 = self.ged_layer3(ged_layer2_d)

        ged_layer0 = ged_layer0.view(b, h // 4, w // 4, -1).contiguous()
        ged_layer1 = ged_layer1.view(b, h // 8, w // 8, -1).contiguous()
        ged_layer2 = ged_layer2.view(b, h // 16, w // 16, -1).contiguous()
        ged_layer3 = ged_layer3.view(b, h // 32, w // 32, -1).contiguous()

        ged_layer0 = ged_layer0.permute(0, 3, 1, 2).contiguous()
        ged_layer1 = ged_layer1.permute(0, 3, 1, 2).contiguous()
        ged_layer2 = ged_layer2.permute(0, 3, 1, 2).contiguous()
        ged_layer3 = ged_layer3.permute(0, 3, 1, 2).contiguous()

        # 同样，在该网络中可删除
        ged_swin_encode_feature4, ged_swin_encode_feature3, ged_swin_encode_feature2, ged_swin_encode_feature1 = ged_layer3, ged_layer2, ged_layer1, ged_layer0
        ged_swin_encode_feature4 = self.ged_swin_encode_up4(ged_swin_encode_feature4)
        ged_swin_encode_feature3 = self.ged_swin_encode_up3(ged_swin_encode_feature3)
        ged_swin_encode_feature2 = self.ged_swin_encode_up2(ged_swin_encode_feature2)
        ged_swin_encode_feature1 = self.ged_swin_encode_up1(ged_swin_encode_feature1)

        layer3_f = ged_layer3
        layer3_f, r1_3, r2_3, per3 = self.GDM3(layer3_f)
        out3 = self.SEM3(r1_3, r2_3)
        s3 = F.interpolate(out3, scale_factor=1/32, mode='bilinear', align_corners=True)
        layer3_f = self.ged_down3(torch.cat((layer3_f, s3), dim=1))
        p32 = self.ged_up_32(layer3_f)

        layer2_f = p32 + ged_layer2
        layer2_f, r1_2, r2_2, per2 = self.GDM2(layer2_f)
        out2 = self.SEM2(r1_2, r2_2)
        s2 = F.interpolate(out2, scale_factor=1/16, mode='bilinear', align_corners=True)
        layer2_f = self.ged_down2(torch.cat((layer2_f, s2), dim=1))
        p21 = self.ged_up_21(layer2_f)

        layer1_f = p21 + ged_layer1
        layer1_f, r1_1, r2_1, per1 = self.GDM1(layer1_f)
        out1 = self.SEM1(r1_1, r2_1)
        s1 = F.interpolate(out1, scale_factor=1/8, mode='bilinear', align_corners=True)
        layer1_f = self.ged_down1(torch.cat((layer1_f, s1), dim=1))
        p10 = self.ged_up_10(layer1_f)

        layer0_f = p10 + ged_layer0
        layer0_f, r1_0, r2_0, per0 = self.GDM0(layer0_f)
        out0 = self.SEM0(r1_0, r2_0)
        # out0_1, out0_2 = torch.chunk(out0.squeeze(0), 2, dim=0)
        # transforms.ToPILImage()(out0_1).show()
        # transforms.ToPILImage()(F.sigmoid(torch.chunk(out0.squeeze(0), 2, dim=0)[0])).show()
        s0 = F.interpolate(out0, scale_factor=1/4, mode='bilinear', align_corners=True)
        layer0_f = self.ged_down0(torch.cat((layer0_f, s0), dim=1))

        ged_layer3_pred = self.ged_pred3(layer3_f)
        ged_layer2_pred = self.ged_pred2(layer2_f)
        ged_layer1_pred = self.ged_pred1(layer1_f)
        ged_layer0_pred = self.ged_pred0(layer0_f)
        ged_layer3_pred = F.upsample(ged_layer3_pred, size=input.size()[2:], mode='bilinear', align_corners=True)
        ged_layer2_pred = F.upsample(ged_layer2_pred, size=input.size()[2:], mode='bilinear', align_corners=True)
        ged_layer1_pred = F.upsample(ged_layer1_pred, size=input.size()[2:], mode='bilinear', align_corners=True)
        ged_layer0_pred = F.upsample(ged_layer0_pred, size=input.size()[2:], mode='bilinear', align_corners=True)

        ged_fuse_feature = torch.cat((input, ged_layer0_pred, ged_layer1_pred, ged_layer2_pred, ged_layer3_pred), dim=1)
        ged_final_pred = self.ghost_final_pred(ged_fuse_feature)

        # GSD
        # ged_swin_encode_feature_fusion = torch.cat((ged_swin_encode_feature4, ged_swin_encode_feature3, ged_swin_encode_feature2, ged_swin_encode_feature1), dim=1)
        # np.save(os.path.join(r"E:\PycharmProjects\GDD\train\gedfusionnpy_new_order", str(self.img_names[self.count][:-4]) + ".npy"), ged_swin_encode_feature_fusion.squeeze(0).cpu())
        # self.count += 1
        ghost = ged_final_pred
        x = torch.cat((input, ghost), dim=1)
        b, c, h, w = x.shape
        x = self.pebed(x)
        x = self.pos_drop(x)

        layer0, layer0_d = self.layer0(x)
        layer1, layer1_d = self.layer1(layer0_d)
        layer2, layer2_d = self.layer2(layer1_d)
        layer3 = self.layer3(layer2_d)

        layer0 = layer0.view(b, h // 4, w // 4, -1).contiguous()
        layer1 = layer1.view(b, h // 8, w // 8, -1).contiguous()
        layer2 = layer2.view(b, h // 16, w // 16, -1).contiguous()
        layer3 = layer3.view(b, h // 32, w // 32, -1).contiguous()

        layer0 = layer0.permute(0, 3, 1, 2).contiguous()
        layer1 = layer1.permute(0, 3, 1, 2).contiguous()
        layer2 = layer2.permute(0, 3, 1, 2).contiguous()
        layer3 = layer3.permute(0, 3, 1, 2).contiguous()

        conv3m_out = layer3  # [1, 1024, 12, 12]

        conv3m_out_ = torch.cat((self.up_32(conv3m_out), layer2), dim=1)
        conv2m_out = self.conv2m(conv3m_out_)  # [1, 512, 24, 24]

        conv2m_out_ = torch.cat((self.up_21(conv2m_out), layer1), dim=1)
        conv1m_out = self.conv1m(conv2m_out_)  # [1, 256, 48, 48]

        conv1m_out_ = torch.cat((self.up_10(conv1m_out), layer0), dim=1)
        conv0m_out = self.conv0m(conv1m_out_)  # [1, 128, 96, 96]

        convfm_out = self.up_final(conv0m_out)  # [1, 64, 384, 384]
        final_pred = self.final_pred(convfm_out)

        layer3_pred = self.pred3(conv3m_out)
        layer2_pred = self.pred2(conv2m_out)
        layer1_pred = self.pred1(conv1m_out)
        layer0_pred = self.pred0(conv0m_out)
        layer3_pred = F.upsample(layer3_pred, size=input.size()[2:], mode='bilinear', align_corners=True)
        layer2_pred = F.upsample(layer2_pred, size=input.size()[2:], mode='bilinear', align_corners=True)
        layer1_pred = F.upsample(layer1_pred, size=input.size()[2:], mode='bilinear', align_corners=True)
        layer0_pred = F.upsample(layer0_pred, size=input.size()[2:], mode='bilinear', align_corners=True)

        h3, w3 = torch.split(out3, [1, 1], 1)
        h2, w2 = torch.split(out2, [1, 1], 1)
        h1, w1 = torch.split(out1, [1, 1], 1)
        h0, w0 = torch.split(out0, [1, 1], 1)
        swinOut = [ged_layer0.detach().cpu().numpy().squeeze().mean(axis=0),
                   ged_layer1.detach().cpu().numpy().squeeze().mean(axis=0),
                   ged_layer2.detach().cpu().numpy().squeeze().mean(axis=0),
                   ged_layer3.detach().cpu().numpy().squeeze().mean(axis=0),
                   layer0.detach().cpu().numpy().squeeze().mean(axis=0),
                   layer1.detach().cpu().numpy().squeeze().mean(axis=0),
                   layer2.detach().cpu().numpy().squeeze().mean(axis=0),
                   layer3.detach().cpu().numpy().squeeze().mean(axis=0)]
        # return F.sigmoid(ged_layer3_pred), F.sigmoid(ged_layer2_pred), F.sigmoid(ged_layer1_pred), F.sigmoid(ged_layer0_pred), F.sigmoid(ged_final_pred)

        return h3, h2, h1, h0, \
               w3, w2, w1, w0, \
               per3, per2, per1, per0, \
               r1_3, r1_2, r1_1, r1_0, \
               r2_3, r2_2, r2_1, r2_0, \
               F.sigmoid(ged_layer3_pred), F.sigmoid(ged_layer2_pred), F.sigmoid(ged_layer1_pred), F.sigmoid(
            ged_layer0_pred), F.sigmoid(ged_final_pred), \
               F.sigmoid(layer3_pred), F.sigmoid(layer2_pred), F.sigmoid(layer1_pred), F.sigmoid(
            layer0_pred), F.sigmoid(final_pred)

# input = torch.randn(2, 3, 384, 384).cuda()
# wrap all things (offset and mask) in DCN
# dcn = Net().cuda()
# output = dcn(input)
# print(output[0].shape)


if __name__ == '__main__':
    import time
    start = time.perf_counter()
    model = Net(backbone_path=r'../backbone/swin_transformer/swin_base_patch4_window12_384.pth').cuda()
    base, head = [], []
    # for name, param in model.named_parameters():
    #     print(name)

    x = torch.randn((1, 3, 384, 384)).cuda()
    flops, params = profile(model, inputs=(x,))
    # print(' Number of parameters:%.4f M' % (params / 1e6))
    # print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))
    end = time.perf_counter()
    print(end - start)


    # x_nonlocal1 = torch.randn((1, 1024, 36, 36)).cuda()
    # x_nonlocal2 = torch.randn((1, 1024, 36, 36)).cuda()
    # NonLocalNet = NLBlockND(in_channels=1024, mode='embedded', dimension=2, bn_layer=True).cuda()
    # flops, params = profile(NonLocalNet, inputs=(x_nonlocal1, x_nonlocal2))
    # print(' Number of parameters:%.4f M' % (params / 1e6))
    # print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))
    #
    # x_dcn1 = torch.randn((1, 1024, 36, 36)).cuda()
    # x_dcn2 = torch.randn((1, 1024, 36, 36)).cuda()
    # DCNNet = DCN_sep(1024, 1024, 3, stride=1, padding=1, dilation=1, deformable_groups=8).cuda()
    # flops, params = profile(DCNNet, inputs=(x_dcn1, x_dcn2))
    # print(' Number of parameters:%.4f M' % (params / 1e6))
    # print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))

    # image_name = "IMG_9279.jpg"
    # image_path_pre = r"E:\PycharmProjects\GhosetNetV3\GEGD\train\image"
    # image_path = os.path.join(image_path_pre, image_name)
    # image = Image.open(image_path).convert('RGB')
    # x = transforms.Compose([
    #     transforms.Resize((384, 384)),
    #     transforms.ToTensor()
    # ])(image).cuda()
    #
    # model = Net(backbone_path=r'../backbone/swin_transformer/swin_base_patch4_window12_384.pth').cuda()
    # model_path = "../logs/Ver.4.28.net2/best.pth"
    # state_dict = torch.load(model_path)
    # model.load_state_dict(state_dict, strict=False)
    # output = model(x.unsqueeze(0))
    # transforms.ToPILImage()(output[23].squeeze(0)).show()
    # print(output[23].shape)
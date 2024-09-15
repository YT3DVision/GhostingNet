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

"""
PAMI重投后，应审稿人意见，将可变性卷积替换成non-local结构，这本质上是尝试将R1掰到R2上，能掰过去则存在双重鬼影，否则不存在。
使用non-local的问题是当只有一个鬼影出现时，它也会被当成两个鬼影，因为non-local的本质并非做平移，单个鬼影相当于位移为0的双重鬼影，
这是从理论层面发现的缺陷，目前难以避免。
Author: huang hao
Date: 2024/3/19
"""

class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, *args):
        """Selectable Attention Mode.
        If '*args' only have a element, self-attention would be used.
        If '*args' only have two elements, then args[0] represents 'q' while args[1] represents 'k_v' and cross-attention would be used.

        Author: huang hao
        Date: 2024/3/18
        """
        assert args.__len__() in [1, 2]

        if args.__len__() == 1:
            return self.forward_self(args[0])
        else:
            return self.forward_cross(args[0], args[1])

    def forward_self(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z

    def forward_cross(self, x, y):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
            y: the same shape as x
        """
        raw_y = y
        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_y = self.g(y).view(batch_size, self.inter_channels, -1)
        g_y = g_y.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_y = y.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_y)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_y = self.phi(y).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_y)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_y = self.phi(y).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_y.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_y = phi_y.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_y], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_y)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection, 'W(k_v)' connected with 'k_v' in this cross attention mode
        z = W_y + raw_y

        return z


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

        # self.dcn = DCN_sep(in_channels, in_channels, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.non_local = NLBlockND(in_channels, mode='embedded', dimension=2, bn_layer=True)

        self.pred1 = nn.Conv2d(self.in_channels, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.pred2 = nn.Conv2d(self.in_channels, 1, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        r1_feature = self.conv1(x)
        r2_feature = self.conv2(x)
        # transforms.ToPILImage()(F.sigmoid(torch.chunk(r1_feature.squeeze(0), 128, dim=0)[127])).show()
        dcn_in = torch.cat((r1_feature, r2_feature), dim=1)
        fea = self.fuse(dcn_in)
        # deform = self.dcn(r1_feature, fea)
        r2_from_nonlocal = self.non_local(r1_feature, r2_feature)

        # per = deform - r2_feature
        per = r2_from_nonlocal - r2_feature

        r1_mask = self.pred1(r1_feature)
        # r2_mask = self.pred2(r2_from_nonlocal + r2_feature)
        r2_mask = self.pred2(r2_from_nonlocal + r2_feature)
        # merge = r1_feature + r2_feature
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
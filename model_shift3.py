import torch.nn.functional as F
from DCNv2.dcn_v2 import *
from collections import OrderedDict
from model_sem import SEM_Net
import torch
import torch.nn as nn

from backbone.swin_transformer.swin_transformer import SwinTransformer_demo


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

        self.dcn = DCN_sep(in_channels, in_channels, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.pred1 = nn.Conv2d(self.in_channels, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.pred2 = nn.Conv2d(self.in_channels, 1, kernel_size=3, stride=1, padding=1, dilation=1)

        self.merge = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        r1_feature = self.conv1(x)
        r2_feature = self.conv2(x)
        dcn_in = torch.cat((r1_feature, r2_feature), dim=1)
        fea = self.fuse(dcn_in)
        deform = self.dcn(r1_feature, fea)
        per = deform - r2_feature

        r1_mask = self.pred1(r1_feature)
        r2_mask = self.pred2(deform + r2_feature)
        merge = r1_feature + r2_feature
        # merge = torch.cat((r1_feature, r2_feature), dim=1)

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


class Net(nn.Module):
    def __init__(self,
                 backbone_path=None,
                 scale=384):
        super(Net, self).__init__()
        self.scale = scale

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

        self.SEM3 = SEM(2, 64)
        self.SEM2 = SEM(2, 64)
        self.SEM1 = SEM(2, 64)
        self.SEM0 = SEM(2, 64)
        

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
        
        self.ged_pred00 = nn.Conv2d(3, 1, 3, 1, 1)
        self.ged_pred11 = nn.Conv2d(3, 1, 3, 1, 1)
        self.ged_pred22 = nn.Conv2d(3, 1, 3, 1, 1)
        self.ged_pred33 = nn.Conv2d(3, 1, 3, 1, 1)
        self.ghost_final_pred = nn.Conv2d(3 + 4, 1, 3, 1, 1)
        self.h_final_pred = nn.Conv2d(4, 1, 3, 1, 1)
        self.w_final_pred = nn.Conv2d(4, 1, 3, 1, 1)
        self.r1_pred = nn.Conv2d(4, 1, 3, 1, 1)
        self.r2_pred = nn.Conv2d(4, 1, 3, 1, 1)

    def forward(self, x):
        # GED
        input = x
        b, c, h, w = x.shape
        x = self.ged_pebed(x)
        x = self.ged_pos_drop(x)

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

        layer3_f = ged_layer3
        layer3_f, r1_3, r2_3, per3 = self.GDM3(layer3_f)
        out3 = self.SEM3(r1_3, r2_3)
        p32 = self.ged_up_32(layer3_f)

        layer2_f = p32 + ged_layer2
        layer2_f, r1_2, r2_2, per2 = self.GDM2(layer2_f)
        out2 = self.SEM2(r1_3, r2_3)
        p21 = self.ged_up_21(layer2_f)

        layer1_f = p21 + ged_layer1
        layer1_f, r1_1, r2_1, per1 = self.GDM1(layer1_f)
        out1 = self.SEM1(r1_3, r2_3)
        p10 = self.ged_up_10(layer1_f)

        layer0_f = p10 + ged_layer0
        layer0_f, r1_0, r2_0, per0 = self.GDM0(layer0_f)
        out0 = self.SEM0(r1_3, r2_3)

        ged_layer3_pred = self.ged_pred3(layer3_f)
        ged_layer2_pred = self.ged_pred2(layer2_f)
        ged_layer1_pred = self.ged_pred1(layer1_f)
        ged_layer0_pred = self.ged_pred0(layer0_f)
        ged_layer3_pred = F.upsample(ged_layer3_pred, size=input.size()[2:], mode='bilinear', align_corners=True)
        ged_layer2_pred = F.upsample(ged_layer2_pred, size=input.size()[2:], mode='bilinear', align_corners=True)
        ged_layer1_pred = F.upsample(ged_layer1_pred, size=input.size()[2:], mode='bilinear', align_corners=True)
        ged_layer0_pred = F.upsample(ged_layer0_pred, size=input.size()[2:], mode='bilinear', align_corners=True)
        # ged_layer3_pred = self.ged_pred33(torch.cat((out3,ged_layer3_pred),dim=1))
        # ged_layer2_pred = self.ged_pred22(torch.cat((out2,ged_layer2_pred),dim=1))
        # ged_layer1_pred = self.ged_pred11(torch.cat((out1,ged_layer1_pred),dim=1))
        # ged_layer0_pred = self.ged_pred00(torch.cat((out0,ged_layer0_pred),dim=1))

        ged_fuse_feature = torch.cat((input, ged_layer0_pred, ged_layer1_pred, ged_layer2_pred, ged_layer3_pred), dim=1)
        ged_final_pred = self.ghost_final_pred(ged_fuse_feature)

        h3, w3 = torch.split(out3, [1, 1], 1)
        h2, w2 = torch.split(out2, [1, 1], 1)
        h1, w1 = torch.split(out1, [1, 1], 1)
        h0, w0 = torch.split(out0, [1, 1], 1)
        
        h = self.h_final_pred(torch.cat((h3, h2, h1, h0), dim=1))
        w = self.w_final_pred(torch.cat((w3, w2, w1, w0), dim=1))
        

        return h3, h2, h1, h0, h,\
               w3, w2, w1, w0, w,\
               per3, per2, per1, per0, \
               r1_3, r1_2, r1_1, r1_0, \
               r2_3, r2_2, r2_1, r2_0, \
               F.sigmoid(ged_layer3_pred), F.sigmoid(ged_layer2_pred), F.sigmoid(ged_layer1_pred), F.sigmoid(
            ged_layer0_pred), F.sigmoid(ged_final_pred)

# input = torch.randn(2, 3, 384, 384).cuda()
# wrap all things (offset and mask) in DCN
# dcn = Net().cuda()
# output = dcn(input)
# print(output[0].shape)

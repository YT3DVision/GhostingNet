import torch.nn.functional as F
from DCNv2.dcn_v2 import *
from collections import OrderedDict
from model_sem import SEM_Net
from torch.autograd import Variable
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
        
    def forward(self, x):
        r1_feature = self.conv1(x)
        r2_feature = self.conv2(x)
        dcn_in = torch.cat((r1_feature, r2_feature), dim=1)
        fea = self.fuse(dcn_in)
        deform = self.dcn(r1_feature, fea)
        per = deform - r2_feature

        r1_mask = self.pred1(r1_feature)
        r2_mask = self.pred2(deform + r2_feature)
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


class Net(nn.Module):
    def __init__(self,
                 backbone_path=None,
                 scale=384):
        super(Net, self).__init__()
        self.scale = scale

        # subnet for glass segmentation
        GSD_swin = SwinTransformer_demo(img_size=384, embed_dim=128, depths=[2, 2, 16, 2],
                                        num_heads=[4, 8, 16, 32], window_size=12)

        if backbone_path is not None:
            state_dict = torch.load(backbone_path)
            pretrained_dict = state_dict['model']
            print("---start load pretrained modle of swin encoder---")
            GSD_swin.load_state_dict(pretrained_dict, strict=False)

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

    def forward(self, x):

        # GSD
        input = x
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

        # return F.sigmoid(ged_layer3_pred), F.sigmoid(ged_layer2_pred), F.sigmoid(ged_layer1_pred), F.sigmoid(ged_layer0_pred), F.sigmoid(ged_final_pred)

        return F.sigmoid(layer3_pred), F.sigmoid(layer2_pred), F.sigmoid(layer1_pred), F.sigmoid(
            layer0_pred), F.sigmoid(final_pred)

# input = torch.randn(2, 3, 384, 384).cuda()
# wrap all things (offset and mask) in DCN
# dcn = Net().cuda()
# output = dcn(input)
# print(output[0].shape)
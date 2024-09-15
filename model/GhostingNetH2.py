import os
from PIL import Image

import torch.nn.functional as F
from DCNv2.dcn_v2 import *
from collections import OrderedDict
from model_sem import SEM_Net
from torch.autograd import Variable
import torch
from torchvision import transforms
import torch.nn as nn

from backbone.swin_transformer.swin_transformer import SwinTransformer_demo


class GhostingNet(nn.Module):
    def __init__(self,
                 backbone_path=r'../backbone/swin_transformer/swin_base_patch4_window12_384.pth',
                 scale=384):
        super().__init__()
        self.scale = scale

        # SwinTransformer backbone
        ged_swin = SwinTransformer_demo(img_size=384, embed_dim=128, depths=[2, 2, 16, 2],
                                        num_heads=[4, 8, 16, 32], window_size=12)
        gsd_swin = SwinTransformer_demo(img_size=384, embed_dim=128, depths=[2, 2, 16, 2],
                                        num_heads=[4, 8, 16, 32], window_size=12)

        if backbone_path is not None:
            state_dict = torch.load(backbone_path)
            pretrained_dict = state_dict['model']
            print("---start load pretrained modle of swin encoder---")
            ged_swin.load_state_dict(pretrained_dict, strict=False)

        self.ged = GED(ged_swin=ged_swin)

        if backbone_path is not None:
            state_dict = torch.load(backbone_path)
            pretrained_dict = state_dict['model']
            gsd_swin.load_state_dict(pretrained_dict, strict=False)
            weight1 = gsd_swin.patch_embed.proj.weight.clone()
            new_first_layer = torch.nn.Conv2d(4, 128, 4, 4)
            # print(new_first_layer.weight.size())
            new_first_layer.weight[:,:3,:,:].data[...] = Variable(weight1, requires_grad=True)
            gsd_swin.patch_embed.proj = new_first_layer

        self.gsd = GSD(gsd_swin=gsd_swin)

    def forward(self, x):
        identity = x

        out4, out3, out2, out1, \
        per4, per3, per2, per1, \
        r1_4, r1_3, r1_2, r1_1, \
        r2_4, r2_3, r2_2, r2_1, \
        ged_layer4_pred, ged_layer3_pred, ged_layer2_pred, ged_layer1_pred, \
        ged_final_pred = self.ged(x)

        # transforms.ToPILImage()(F.sigmoid(ged_final_pred.squeeze(0))).show()
        transforms.ToPILImage()(ged_final_pred.squeeze(0)).show()

        h4, w4 = torch.split(out4, [1, 1], 1)
        h3, w3 = torch.split(out3, [1, 1], 1)
        h2, w2 = torch.split(out2, [1, 1], 1)
        h1, w1 = torch.split(out1, [1, 1], 1)

        layer4_pred, \
        layer3_pred, \
        layer2_pred, \
        layer1_pred, \
        final_pred = self.gsd(identity, ged_final_pred)

        return h4, h3, h2, h1, \
               w4, w3, w2, w1, \
               per4, per3, per2, per1, \
               r1_4, r1_3, r1_2, r1_1, \
               r2_4, r2_3, r2_2, r2_1, \
               F.sigmoid(ged_layer4_pred), F.sigmoid(ged_layer3_pred), F.sigmoid(ged_layer2_pred), F.sigmoid(ged_layer1_pred), F.sigmoid(ged_final_pred), \
               F.sigmoid(layer4_pred), F.sigmoid(layer3_pred), F.sigmoid(layer2_pred), F.sigmoid(layer1_pred), F.sigmoid(final_pred)


class GED(nn.Module):
    """
    提取图片鬼影特征，获取图片鬼影的mask
    """
    def __init__(self, ged_swin):
        super().__init__()

        self.ged_pebed = ged_swin.patch_embed
        self.ged_pos_drop = ged_swin.pos_drop
        self.ged_layer1 = ged_swin.layers[0]
        self.ged_layer2 = ged_swin.layers[1]
        self.ged_layer3 = ged_swin.layers[2]
        self.ged_layer4 = ged_swin.layers[3]

        self.dre4 = DRE(1024)
        self.dre3 = DRE(512)
        self.dre2 = DRE(256)
        self.dre1 = DRE(128)

        self.sem4 = SEM(2, 64)
        self.sem3 = SEM(2, 64)
        self.sem2 = SEM(2, 64)
        self.sem1 = SEM(2, 64)

        self.ged_pred0 = nn.Conv2d(128, 1, 3, 1, 1)
        self.ged_pred1 = nn.Conv2d(256, 1, 3, 1, 1)
        self.ged_pred2 = nn.Conv2d(512, 1, 3, 1, 1)
        self.ged_pred3 = nn.Conv2d(1024, 1, 3, 1, 1)
        self.ghost_final_pred = nn.Conv2d(3 + 4, 1, 3, 1, 1)

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

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x = self.ged_pebed(x)
        x = self.ged_pos_drop(x)

        ged_layer1, ged_layer0_d = self.ged_layer1(x)
        ged_layer2, ged_layer1_d = self.ged_layer2(ged_layer0_d)
        ged_layer3, ged_layer2_d = self.ged_layer3(ged_layer1_d)
        ged_layer4 = self.ged_layer4(ged_layer2_d)

        ged_layer1 = ged_layer1.view(b, h // 4, w // 4, -1).contiguous()
        ged_layer2 = ged_layer2.view(b, h // 8, w // 8, -1).contiguous()
        ged_layer3 = ged_layer3.view(b, h // 16, w // 16, -1).contiguous()
        ged_layer4 = ged_layer4.view(b, h // 32, w // 32, -1).contiguous()

        ged_layer1 = ged_layer1.permute(0, 3, 1, 2).contiguous()
        ged_layer2 = ged_layer2.permute(0, 3, 1, 2).contiguous()
        ged_layer3 = ged_layer3.permute(0, 3, 1, 2).contiguous()
        ged_layer4 = ged_layer4.permute(0, 3, 1, 2).contiguous()

        layer4_f = ged_layer4
        layer4_f, r1_4, r2_4, per4 = self.dre4(layer4_f)
        out4 = self.sem4(r1_4, r2_4)
        shift3 = F.interpolate(out4, scale_factor=1 / 32, mode='bilinear', align_corners=True)
        layer4_f = self.ged_down3(torch.cat((layer4_f, shift3), dim=1))
        p32 = self.ged_up_32(layer4_f)

        layer3_f = p32 + ged_layer3
        layer3_f, r1_3, r2_3, per3 = self.dre3(layer3_f)
        out3 = self.sem3(r1_3, r2_3)
        shift2 = F.interpolate(out3, scale_factor=1 / 16, mode='bilinear', align_corners=True)
        layer3_f = self.ged_down2(torch.cat((layer3_f, shift2), dim=1))
        p21 = self.ged_up_21(layer3_f)

        layer2_f = p21 + ged_layer2
        layer2_f, r1_2, r2_2, per2 = self.dre2(layer2_f)
        out2 = self.sem2(r1_2, r2_2)
        shift1 = F.interpolate(out2, scale_factor=1 / 8, mode='bilinear', align_corners=True)
        layer2_f = self.ged_down1(torch.cat((layer2_f, shift1), dim=1))
        p10 = self.ged_up_10(layer2_f)

        layer1_f = p10 + ged_layer1
        layer1_f, r1_1, r2_1, per1 = self.dre1(layer1_f)
        out1 = self.sem1(r1_1, r2_1)
        shift0 = F.interpolate(out1, scale_factor=1 / 4, mode='bilinear', align_corners=True)
        layer1_f = self.ged_down0(torch.cat((layer1_f, shift0), dim=1))

        ged_layer4_pred = self.ged_pred3(layer4_f)
        ged_layer3_pred = self.ged_pred2(layer3_f)
        ged_layer2_pred = self.ged_pred1(layer2_f)
        ged_layer1_pred = self.ged_pred0(layer1_f)
        ged_layer4_pred = F.upsample(ged_layer4_pred, size=identity.size()[2:], mode='bilinear', align_corners=True)
        ged_layer3_pred = F.upsample(ged_layer3_pred, size=identity.size()[2:], mode='bilinear', align_corners=True)
        ged_layer2_pred = F.upsample(ged_layer2_pred, size=identity.size()[2:], mode='bilinear', align_corners=True)
        ged_layer1_pred = F.upsample(ged_layer1_pred, size=identity.size()[2:], mode='bilinear', align_corners=True)

        ged_fuse_feature = torch.cat((identity, ged_layer1_pred, ged_layer2_pred, ged_layer3_pred, ged_layer4_pred), dim=1)
        ged_final_pred = self.ghost_final_pred(ged_fuse_feature)

        return out4, out3, out2, out1, \
               per4, per3, per2, per1, \
               r1_4, r1_3, r1_2, r1_1, \
               r2_4, r2_3, r2_2, r2_1, \
               ged_layer4_pred, ged_layer3_pred, ged_layer2_pred, ged_layer1_pred, \
               ged_final_pred


class GSD(nn.Module):
    """
    使用SwinTransformer做玻璃分割
    """
    def __init__(self, gsd_swin):
        super().__init__()

        self.pebed = gsd_swin.patch_embed
        self.pos_drop = gsd_swin.pos_drop
        self.layer1 = gsd_swin.layers[0]
        self.layer2 = gsd_swin.layers[1]
        self.layer3 = gsd_swin.layers[2]
        self.layer4 = gsd_swin.layers[3]

        self.up_33 = self.upsample(1024, 512)
        self.up_22 = self.upsample(512, 256)
        self.up_11 = self.upsample(256, 128)
        self.up_final = self.upsample(128, 64, kernel_size=8, stride=4, padding=2)

        self.conv3m = MLP(1024, 512, useBN=True)
        self.conv2m = MLP(512, 256, useBN=True)
        self.conv1m = MLP(256, 128, useBN=True)

        self.final_pred = nn.Conv2d(64, 1, 3, 1, 1)

        self.pred1 = nn.Conv2d(128, 1, 3, 1, 1)
        self.pred2 = nn.Conv2d(256, 1, 3, 1, 1)
        self.pred3 = nn.Conv2d(512, 1, 3, 1, 1)
        self.pred4 = nn.Conv2d(1024, 1, 3, 1, 1)

    def upsample(self, ch_coarse, ch_fine, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, kernel_size, stride, padding, bias=False),
            nn.ReLU()
        )

    def forward(self, x, ghost):
        identity = x
        x = torch.cat((identity, ghost), dim=1)
        b, c, h, w = x.shape
        x = self.pebed(x)
        x = self.pos_drop(x)

        layer1, layer1_d = self.layer1(x)
        layer2, layer2_d = self.layer2(layer1_d)
        layer3, layer3_d = self.layer3(layer2_d)
        layer4 = self.layer4(layer3_d)

        layer1 = layer1.view(b, h // 4, w // 4, -1).contiguous()
        layer2 = layer2.view(b, h // 8, w // 8, -1).contiguous()
        layer3 = layer3.view(b, h // 16, w // 16, -1).contiguous()
        layer4 = layer4.view(b, h // 32, w // 32, -1).contiguous()

        layer1 = layer1.permute(0, 3, 1, 2).contiguous()
        layer2 = layer2.permute(0, 3, 1, 2).contiguous()
        layer3 = layer3.permute(0, 3, 1, 2).contiguous()
        layer4 = layer4.permute(0, 3, 1, 2).contiguous()

        conv4m_out = layer4  # [1, 1024, 12, 12]

        conv3m_out_ = torch.cat((self.up_33(conv4m_out), layer3), dim=1)
        conv3m_out = self.conv3m(conv3m_out_)  # [1, 512, 24, 24]

        conv2m_out_ = torch.cat((self.up_22(conv3m_out), layer2), dim=1)
        conv2m_out = self.conv2m(conv2m_out_)  # [1, 256, 48, 48]

        conv1m_out_ = torch.cat((self.up_11(conv2m_out), layer1), dim=1)
        conv1m_out = self.conv1m(conv1m_out_)  # [1, 128, 96, 96]

        convfm_out = self.up_final(conv1m_out)  # [1, 64, 384, 384]
        final_pred = self.final_pred(convfm_out)

        layer4_pred = self.pred4(conv4m_out)
        layer3_pred = self.pred3(conv3m_out)
        layer2_pred = self.pred2(conv2m_out)
        layer1_pred = self.pred1(conv1m_out)
        layer4_pred = F.upsample(layer4_pred, size=identity.size()[2:], mode='bilinear', align_corners=True)
        layer3_pred = F.upsample(layer3_pred, size=identity.size()[2:], mode='bilinear', align_corners=True)
        layer2_pred = F.upsample(layer2_pred, size=identity.size()[2:], mode='bilinear', align_corners=True)
        layer1_pred = F.upsample(layer1_pred, size=identity.size()[2:], mode='bilinear', align_corners=True)

        return layer4_pred, layer3_pred, layer2_pred, layer1_pred, final_pred


class DRE(nn.Module):
    def __init__(self, in_channels, scale=384, groups=8):
        super(DRE, self).__init__()

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
        super().__init__()
        # ---------------- Encoder ----------------------
        self.conv1 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(inc_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # ---------------- Decoder ----------------------
        self.deconv5 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
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

        self.deconv1 = nn.Conv2d(64, 2, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, r1, r2):
        # ---------------- Encoder ----------------------
        hx = torch.cat((r1, r2), dim=1)
        hx = self.conv1(hx)

        hx1 = self.conv2(hx)
        hx = self.pool1(hx1)

        hx2 = self.conv3(hx)
        hx = self.pool2(hx2)

        hx3 = self.conv4(hx)
        hx = self.pool3(hx3)

        hx4 = self.conv5(hx)
        hx = self.pool4(hx4)

        hx5 = self.conv6(hx)

        # ---------------- Decoder ----------------------
        hx = self.upsample(hx5)

        d4 = self.deconv5(torch.cat((hx, hx4), 1))
        hx = self.upsample(d4)

        d3 = self.deconv4(torch.cat((hx, hx3), 1))
        hx = self.upsample(d3)

        d2 = self.deconv3(torch.cat((hx, hx2), 1))
        hx = self.upsample(d2)

        d1 = self.deconv2(torch.cat((hx, hx1), 1))
        output = self.deconv1(d1)

        # output = self.fuse(torch.cat((ref, mask), 1))

        # h, w = torch.split(output, [1, 1], 1)
        return output


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU()
            # nn.LeakyReLU(0.1)
        ) if useBN else nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    image_name = "IMG_9279.jpg"
    image_path_pre = r"E:\PycharmProjects\GhosetNetV3\GEGD\train\image"
    image_path = os.path.join(image_path_pre, image_name)
    image = Image.open(image_path).convert('RGB')
    x = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])(image).cuda()

    model = GhostingNet().cuda()
    model_path = "../logs/Ver.4.28.net2/best.pth"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    output = model(x.unsqueeze(0))
    print(output[0].shape)

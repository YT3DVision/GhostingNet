import torch
import torch.nn as nn
import torch.nn.functional as F

from DCNv2.gjh.backbone.swin_transformer.swin_transformer import SwinTransformer_demo


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
    def __init__(self, in_channels, scale=384):
        super(GDM, self).__init__()

        self.in_channels = in_channels
        self.scale = scale

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.pred1 = nn.Conv2d(self.in_channels, 1, kernel_size=3, stride=1, padding=1, dilation=1)
        self.pred2 = nn.Conv2d(self.in_channels, 1, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        r1_feature = self.conv1(x)
        r2_feature = self.conv2(x)
        ghost_feature = r1_feature + r2_feature

        r1_mask = self.pred1(r1_feature)
        r2_mask = self.pred2(r2_feature)
        r1_mask = F.upsample(r1_mask, size=[self.scale, self.scale], mode='bilinear', align_corners=True)
        r2_mask = F.upsample(r2_mask, size=[self.scale, self.scale], mode='bilinear', align_corners=True)

        return ghost_feature, r1_mask, r2_mask


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
        self.deconv4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.deconv0 = nn.Conv2d(64, 2, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, r1, r2):
        # ---------------- Encoder ----------------------
        hx = torch.cat((x, r1, r2), dim=1)
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
    def __init__(self, backbone_path=None, pred_channel=48):
        super(Net, self).__init__()

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

        self.SEM = SEM(5, 64)

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

        self.ged_fuse2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.ged_fuse1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.ged_fuse0 = nn.Conv2d(128, 128, 3, 1, 1)

        self.ged_pred0 = nn.Conv2d(128, 1, 3, 1, 1)
        self.ged_pred1 = nn.Conv2d(256, 1, 3, 1, 1)
        self.ged_pred2 = nn.Conv2d(512, 1, 3, 1, 1)
        self.ged_pred3 = nn.Conv2d(1024, 1, 3, 1, 1)
        self.ghost_final_pred = nn.Conv2d(3 + 4, 1, 3, 1, 1)
        self.r1_pred = nn.Conv2d(4, 1, 3, 1, 1)
        self.r2_pred = nn.Conv2d(4, 1, 3, 1, 1)

        # subnet for glass segmentation
        GSD_swin = SwinTransformer_demo(img_size=384, embed_dim=128, depths=[2, 2, 2, 2],
                                        num_heads=[4, 8, 16, 32], window_size=12)

        self.first_conv = nn.Conv2d(4, 3, 3, 1, 1)

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
        # GED
        input = x
        b, c, h, w = x.shape
        x = self.ged_pebed(x)
        x = self.ged_pos_drop(x)

        ged_layer0, ged_layer0_d = self.ged_layer0(x)
        ged_layer1, ged_layer1_d = self.ged_layer1(ged_layer0_d)
        ged_layer2, ged_layer2_d = self.ged_layer2(ged_layer1_d)
        ged_layer3 = self.ged_layer3(ged_layer2_d)

        ged_layer0 = ged_layer0.view(b, h // 4, w // 4, -1)
        ged_layer1 = ged_layer1.view(b, h // 8, w // 8, -1)
        ged_layer2 = ged_layer2.view(b, h // 16, w // 16, -1)
        ged_layer3 = ged_layer3.view(b, h // 32, w // 32, -1)

        ged_layer0 = ged_layer0.permute(0, 3, 1, 2)
        ged_layer1 = ged_layer1.permute(0, 3, 1, 2)
        ged_layer2 = ged_layer2.permute(0, 3, 1, 2)
        ged_layer3 = ged_layer3.permute(0, 3, 1, 2)

        layer3_f = ged_layer3
        layer3_f, r1_3, r2_3 = self.GDM3(layer3_f)
        p32 = self.ged_up_32(layer3_f)

        layer2_f = p32 + ged_layer2
        layer2_f, r1_2, r2_2 = self.GDM2(layer2_f)
        # layer2_f = self.ged_fuse2(layer2_f)
        p21 = self.ged_up_21(layer2_f)

        layer1_f = p21 + ged_layer1
        layer1_f, r1_1, r2_1 = self.GDM1(layer1_f)
        # layer1_f = self.ged_fuse1(layer1_f)
        p10 = self.ged_up_10(layer1_f)

        layer0_f = p10 + ged_layer0
        layer0_f, r1_0, r2_0 = self.GDM0(layer0_f)
        # layer0_f = self.ged_fuse0(layer0_f)

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

        r1 = self.r1_pred(torch.cat((r1_3, r1_2, r1_1, r1_0), dim=1))
        r2 = self.r2_pred(torch.cat((r2_3, r2_2, r2_1, r2_0), dim=1))

        out = self.SEM(input, r1, r2)
        # print(out.size())
        out_abs = torch.abs(out)
        # print(out_abs.size())
        out_sum = out.sum(dim=1)
        # print(out_sum.size())
        out_usq = out_sum.unsqueeze(dim=1)
        # print(out_usq.size())
        ged_final_pred = ged_final_pred * out_usq
        # print(ged_final_pred.size())
        # shift_abs = torch.abs(h) + torch.abs(w)
        # ged_final_pred = ged_final_pred * shift_abs
        # print(ged_final_pred.size())

        # GSD
        ghost = ged_final_pred
        x = torch.cat((input, ghost), dim=1)
        x = self.first_conv(x)
        x = self.pebed(x)
        x = self.pos_drop(x)

        layer0, layer0_d = self.layer0(x)
        layer1, layer1_d = self.layer1(layer0_d)
        layer2, layer2_d = self.layer2(layer1_d)
        layer3 = self.layer3(layer2_d)

        layer0 = layer0.view(b, h // 4, w // 4, -1)
        layer1 = layer1.view(b, h // 8, w // 8, -1)
        layer2 = layer2.view(b, h // 16, w // 16, -1)
        layer3 = layer3.view(b, h // 32, w // 32, -1)

        layer0 = layer0.permute(0, 3, 1, 2)
        layer1 = layer1.permute(0, 3, 1, 2)
        layer2 = layer2.permute(0, 3, 1, 2)
        layer3 = layer3.permute(0, 3, 1, 2)

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
        
        h, w = torch.split(out, [1, 1], 1)

        return h, w,\
               F.sigmoid(r1_3), F.sigmoid(r1_2), F.sigmoid(r1_1), F.sigmoid(r1_0), F.sigmoid(r1), \
               F.sigmoid(r2_3), F.sigmoid(r2_2), F.sigmoid(r2_1), F.sigmoid(r2_0), F.sigmoid(r2), \
               F.sigmoid(ged_layer3_pred), F.sigmoid(ged_layer2_pred), F.sigmoid(ged_layer1_pred), F.sigmoid(
            ged_layer0_pred), F.sigmoid(ged_final_pred), \
               F.sigmoid(layer3_pred), F.sigmoid(layer2_pred), F.sigmoid(layer1_pred), F.sigmoid(
            layer0_pred), F.sigmoid(final_pred)
            

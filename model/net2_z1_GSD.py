import torch.nn.functional as F
from DCNv2.dcn_v2 import *
from collections import OrderedDict
from model_sem import SEM_Net
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
from backbone.swin_transformer.swin_transformer import SwinTransformer_demo
import warnings

warnings.filterwarnings("ignore")


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


class UpSampleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, [384, 384], mode='bilinear', align_corners=True)


class Encoder_Fea_Fusion(nn.Module):
    def __init__(self, dim):
        super(Encoder_Fea_Fusion, self).__init__()
        self.reduction = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x, y):
        fea = torch.cat([x, y], dim=-1)
        out = self.reduction(fea)
        return out


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
            GSD_swin.load_state_dict(pretrained_dict, strict=False)
            weight1 = GSD_swin.patch_embed.proj.weight.clone()

            new_first_layer = torch.nn.Conv2d(4, 128, 4, 4)
            # new_first_layer = torch.nn.Conv2d(4, 128, 4, 4)
            new_first_layer.weight[:, :3, :, :].data[...] = Variable(weight1, requires_grad=True)
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

        self.ged_swin_encode_up4 = UpSampleBlock(1024)
        self.ged_swin_encode_up3 = UpSampleBlock(512)
        self.ged_swin_encode_up2 = UpSampleBlock(256)
        self.ged_swin_encode_up1 = UpSampleBlock(128)

        self.fusion0 = Encoder_Fea_Fusion(dim=128)
        self.fusion1 = Encoder_Fea_Fusion(dim=256)
        self.fusion2 = Encoder_Fea_Fusion(dim=512)
        self.fusion3 = Encoder_Fea_Fusion(dim=1024)

    def forward(self, x, ghost, gscf0, gscf1, gscf2, gscf3):
        identify = x

        x = torch.cat((identify, ghost), dim=1)
        b, c, h, w = x.shape
        x = self.pebed(x)
        x = self.pos_drop(x)

        f0 = self.fusion0(x, gscf0)
        layer0, layer0_d = self.layer0(f0)

        f1 = self.fusion1(layer0_d, gscf1)
        layer1, layer1_d = self.layer1(f1)

        f2 = self.fusion2(layer1_d, gscf2)
        layer2, layer2_d = self.layer2(f2)

        f3 = self.fusion3(layer2_d, gscf3)
        layer3 = self.layer3(f3)

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
        layer3_pred = F.upsample(layer3_pred, size=identify.size()[2:], mode='bilinear', align_corners=True)
        layer2_pred = F.upsample(layer2_pred, size=identify.size()[2:], mode='bilinear', align_corners=True)
        layer1_pred = F.upsample(layer1_pred, size=identify.size()[2:], mode='bilinear', align_corners=True)
        layer0_pred = F.upsample(layer0_pred, size=identify.size()[2:], mode='bilinear', align_corners=True)

        return F.sigmoid(layer3_pred), \
               F.sigmoid(layer2_pred), \
               F.sigmoid(layer1_pred),\
               F.sigmoid(layer0_pred), \
               F.sigmoid(final_pred), \


    def _get_name(self):
        return "net2_z1_GSD"


if __name__ == '__main__':
    image_name = "IMG_9279.jpg"
    image_path_pre = r"E:\PycharmProjects\GhosetNetV3\GEGD\train\image"
    image_path = os.path.join(image_path_pre, image_name)
    image = Image.open(image_path).convert('RGB')
    x = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])(image).cuda()

    model = Net(backbone_path=r'../backbone/swin_transformer/swin_base_patch4_window12_384.pth').cuda()
    # model_path = "../logs/Ver.4.28.net2/best.pth"
    # state_dict = torch.load(model_path)
    # model.load_state_dict(state_dict, strict=False)
    output = model(x.unsqueeze(0))
    print(output[29].shape)

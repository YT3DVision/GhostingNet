from metrics import *
import os
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import torch

# GEGD GHOST
# gt_dir = 'D:/GhostDataSet/GEGD_eval_ghost'
gt_dir = r'E:\PycharmProjects\GhosetNetV3\GEGD\test'
# pred_dir = 'G:/GEGD_ghost_eval/wait'
# pred_dir = 'G:/contrast_ghost2/finished/swin_57.01'
pred_dir = r'E:\PycharmProjects\GhosetNetV3\Run_hh_recurrent'

from utils.Miou import iou_mean

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

scale = 384
img_transform = transforms.Compose([
    transforms.Resize((scale, scale)),
    transforms.ToTensor()
])


for sub_dir in os.listdir(pred_dir):
    v_glass_iou = 0
    # print(len(os.listdir(os.path.join(pred_dir, sub_dir))))
    if 0 == len(os.listdir(os.path.join(pred_dir, sub_dir))):
        continue

    for name in os.listdir(os.path.join(pred_dir, sub_dir)):
        # print(name)
        if name.endswith(".npy"):
            continue
        gt = Image.open(os.path.join(gt_dir, sub_dir, name))
        gt_var = Variable(img_transform(gt).unsqueeze(0)).cuda()
        pred = Image.open(os.path.join(pred_dir, sub_dir, name))
        pred_var = Variable(img_transform(pred).unsqueeze(0)).cuda()

        a = gt_var
        b = pred_var
        a = torch.round(a).squeeze(0).int().detach().cpu()
        b = torch.round(b).squeeze(0).int().detach().cpu()
        # print(iou_mean(a, b, 1))
        v_glass_iou += iou_mean(a, b, 1)
        # print(v_glass_iou)
    print(sub_dir)
    print(v_glass_iou/len(os.listdir(os.path.join(pred_dir, sub_dir))))
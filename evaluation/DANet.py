import argparse
import os
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.autograd import Variable
from compare_methods.mirrornet import MirrorNet

from config import *
from torchvision import transforms
from utils.Miou import iou_mean

# from model.Jiaying import GlassNet

# from MacroNet.focalloader import Focal
from datasets.lfm import LFM

import numpy as np
from PIL import Image

w, h = 540, 375


def main():
    #
    parser = argparse.ArgumentParser(description="PyTorch Mirror Detection Example")
    parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')

    # dataset settings
    parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
    parser.add_argument("--workers", type=int, default=1, help="DataLoader Num workers")
    parser.add_argument("--pin_memory", type=bool, default=False, help="DataLoader pin memory")

    parser.add_argument("--use_GPU", type=bool, default=True, help='')
    parser.add_argument("--data_path", type=str, default="./LFM_V2/", help='Path to LFM')
    parser.add_argument("--save_path", type=str, default="./compare/MirrorNet", help='')
    parser.add_argument("--result_path", type=str, default="./results/MirrorNet", help='')  # 88.95
    # parser.add_argument('--useBN', action='store_true', help='enalbes batch normalization')

    parser.add_argument("--angRes", type=int, default=9, help='')

    opt = parser.parse_args()

    # train/val list
    train_list = [f for f in os.listdir(os.path.join(opt.data_path, 'Train', 'train_cv'))]
    val_list = [f for f in os.listdir(os.path.join(opt.data_path, 'Test', 'test_cv'))]

    # Load Data
    lfm_train = LFM(location=opt.data_path, train=True)
    lfm_val = LFM(location=opt.data_path, train=False)
    train_loader = DataLoader(
        dataset=lfm_train,
        batch_size=opt.batchSize,
        num_workers=opt.workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )
    val_loader = DataLoader(
        dataset=lfm_val,
        batch_size=opt.batchSize,
        num_workers=opt.workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )
    print("INFO: training samples: %d\n" % int(len(train_list)))
    print("INFO: valid samples: %d\n" % int(len(val_list)))

    if opt.use_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    to_pil = transforms.ToPILImage()

    if not os.path.isdir(os.path.join(opt.result_path, 'mask')):
        os.makedirs(os.path.join(opt.result_path, 'mask'))

    net = MirrorNet(backbone_path=None)

    if opt.use_GPU:
        net = net.cuda()

    net.load_state_dict(torch.load(os.path.join(opt.save_path, 'iou.pth')))
    net.eval()
    with torch.no_grad():

        for i, (cv, ms, depth, mask, names) in enumerate(val_loader):
            # measure data loading time
            cv = Variable(cv)

            cv = cv.cuda()

            p4, p3, p2, p1 = net(cv)

            p = p1
            p = p.data.squeeze(0)
            p = np.array(transforms.Resize((h, w))(to_pil(p)))
            Image.fromarray(p).save(os.path.join(opt.result_path, 'mask', names[0]))

            # print(p)
            print(names[0])


if __name__ == '__main__':
    main()

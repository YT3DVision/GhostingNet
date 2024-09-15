import argparse
import os
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.autograd import Variable
from config import *
from torchvision import transforms
from utils.Miou import iou_mean
from utils.loss import bce_ssim_loss

from compare_methods.mirrornet import MirrorNet

# from MacroNet.focalloader import Focal
from datasets.lfm import LFM
from datasets.dutlf_v2_focal_mm import Focal


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def main():
    train_min_loss = float('inf')
    valid_min_loss = float('inf')
    valid_max_glass = float(0)
    valid_max_ghost = float(0)

    parser = argparse.ArgumentParser(description='LFSalient Training')

    # dataset settings
    parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
    parser.add_argument("--workers", type=int, default=16, help="DataLoader Num workers")

    # optimizer settings
    parser.add_argument("--lr", type=float, default=1e-5, help='learning rate')
    parser.add_argument("--wd", type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--milestones', default=[30, 60, 90, 120, 150, 180], type=int, help='Learning rate decay steps')
    parser.add_argument('--clip', default=0.5, type=float, help='Gradient clip')

    # training settings
    parser.add_argument("--epoch", type=int, default=400, help='')
    parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
    parser.add_argument("--use_GPU", type=bool, default=True, help='')
    # parser.add_argument("--data_path", type=str, default="/data/Generation/DUTLF-V2/", help='Path to DUTLF_V2')  # synthetic
    parser.add_argument("--data_path", type=str, default="./LFM_V2/", help='Path to LFM')  # synthetic
    # parser.add_argument("--data_path", type=str, default="D:/.jh_code/DUTLF-V2/", help='Path to DUTLF_V2')  # synthetic
    # parser.add_argument("--model_path", type=str, default="/data/gjhLF/logs/Focal_VGG_dac_filp_selayer/iou.pth", help='')
    # parser.add_argument("--sav e_path", type=str, default="/data/gjhLF/logs
    # /Focal_VGG_dac_filp_selayer200-400", help='')
    # parser.add_argument("--save_path", type=str, default="D:/.00_code/logs/cv_depth_res101_conv_1e-6-2", help='')
    parser.add_argument("--model_path", type=str, default="./compare/MirrorNet ", help='')
    parser.add_argument("--save_path", type=str, default="./compare/MirrorNet", help='')
    parser.add_argument("--log_path", type=str, default="./compare/MirrorNet.log", help='')
    # af2--? depth-mirror-cbam
    parser.add_argument("--scale", type=int, default=256, help='')
    parser.add_argument("--angRes", type=int, default=9, help='')
    parser.add_argument("--backbone", type=str, default="res101", help='')
    # parser.add_argument("--momentum", type=float, default=0.9, help='')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # train/val list
    train_list = [f for f in os.listdir(os.path.join(args.data_path, 'Train', 'train_cv'))]
    val_list = [f for f in os.listdir(os.path.join(args.data_path, 'Test', 'test_cv'))]

    # Load Data
    lfm_train = LFM(location=args.data_path, train=True)
    lfm_val = LFM(location=args.data_path, train=False)
    train_loader = DataLoader(
        dataset=lfm_train,
        batch_size=args.batchSize,
        num_workers=args.workers,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=lfm_val,
        batch_size=args.batchSize,
        num_workers=args.workers,
        shuffle=False
    )
    print("INFO: training samples: %d\n" % int(len(train_list)))
    print("INFO: valid samples: %d\n" % int(len(val_list)))

    # model
    if args.backbone == 'res':
        model = Net(angRes=args.angRes, backbone_path=None)
    elif args.backbone == 'res101':
        net = MirrorNet(backbone_path=ResNeXt_path)
        # model.load_state_dict(torch.load(os.path.join(args.save_path, 'iou.pth')))
    elif args.backbone == 'res':
        model = Net(angRes=args.angRes, pretrained=True)
    elif args.backbone == 'swint':
        model = Net(angRes=args.angRes, backbone_path=swin_T)
    elif args.backbone == 'swinb':
        model = Net(angRes=args.angRes, backbone_path=swin_B)
    elif args.backbone == 'res50':
        model = Net(backbone_path=Res50_path, angRes=args.angRes, pretrained=True)

    '''
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict, strict=False)
    '''

    if args.use_GPU:
        net = net.cuda()

    # loss
    criterion = nn.BCELoss()
    criterion = criterion.cuda()

    # optimizer
    print("---optimizer...")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)

    print("---Info: Start Training...")
    for epoch in range(args.epoch):

        ite_num = 0
        ite_num4val = 0
        running_loss = 0.0
        running_val_loss = 0.0
        t_iou = 0

        net.train()
        net.zero_grad()
        start = time.time()

        for i, (cv, ms, depth, mask, names) in enumerate(train_loader):
            ite_num = ite_num + 1

            # measure data loading time
            cv = Variable(cv)
            ms = [Variable(mm) for mm in ms]
            mask = Variable(mask)
            depth = Variable(depth)

            cv = cv.cuda()
            ms = [mm.cuda() for mm in ms]
            mask = mask.cuda()
            depth = depth.cuda()

            p4, p3, p2, p1 = net(cv)

            loss4 = bce_ssim_loss(p4, mask)
            loss3 = bce_ssim_loss(p3, mask)
            loss2 = bce_ssim_loss(p2, mask)
            loss1 = bce_ssim_loss(p1, mask)

            loss = loss4 + 2 * loss3 + 3 * loss2 + 4 * loss1
            loss.backward()
            # clip_gradient(optimizer, args.clip)
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # calculate iou
            pred = p1
            mask = mask
            bs, _, _, _ = mask.shape

            temp1 = pred.data.squeeze(1)
            temp2 = mask.data.squeeze(1)
            for i in range(bs):
                a = temp1[i, :, :]
                b = temp2[i, :, :]
                a = torch.round(a).squeeze(0).int().detach().cpu()
                b = torch.round(b).squeeze(0).int().detach().cpu()
                t_iou += iou_mean(a, b, 1)

            '''
            print("[epoch: %3d/%3d, batch: %5d/2957, ite: %d] train loss: %3f " % (
                epoch + 1, args.epoch, (i + 1) * args.batchSize, ite_num, running_loss / ite_num))
            '''
        # scheduler.step()

        net.eval()
        net.zero_grad()

        v_iou = 0

        with torch.no_grad():
            for i, (cv, ms, depth, mask, names) in enumerate(val_loader):
                ite_num4val = ite_num4val + 1

                # measure data loading time
                cv = Variable(cv)
                ms = [Variable(mm) for mm in ms]
                mask = Variable(mask)
                depth = Variable(depth)

                cv = cv.cuda()
                ms = [mm.cuda() for mm in ms]
                mask = mask.cuda()
                depth = depth.cuda()

                p4, p3, p2, p1 = net(cv)

                loss4 = bce_ssim_loss(p4, mask)
                loss3 = bce_ssim_loss(p3, mask)
                loss2 = bce_ssim_loss(p2, mask)
                loss1 = bce_ssim_loss(p1, mask)

                vloss = loss4 + 2 * loss3 + 3 * loss2 + 4 * loss1

                running_val_loss += vloss.item()

                # calculate iou
                pred = p1
                mask = mask
                bs, _, _, _ = mask.shape

                temp1 = pred.data.squeeze(1)
                temp2 = mask.data.squeeze(1)
                for i in range(bs):
                    a = temp1[i, :, :]
                    b = temp2[i, :, :]
                    a = torch.round(a).squeeze(0).int().detach().cpu()
                    b = torch.round(b).squeeze(0).int().detach().cpu()
                    v_iou += iou_mean(a, b, 1)

        end = time.time()
        t = end - start

        print("[epoch: %3d/%3d] train loss: %5f  valid loss: %5f valid iou: %.6f total_time: %5.2f" % (
            epoch + 1, args.epoch, running_loss / len(train_loader) / args.batchSize,
            running_val_loss / len(val_loader) / args.batchSize, v_iou / len(val_list), t))
        print("train iou: %.6f" % (t_iou / len(train_list)))

        mylog = open(args.log_path, mode='a', encoding='utf-8')

        print("[epoch: %3d/%3d] train loss: %5f  valid loss: %5f valid iou: %.6f total_time: %5.2f" % (
            epoch + 1, args.epoch, running_loss / len(train_loader) / args.batchSize,
            running_val_loss / len(val_loader) / args.batchSize, v_iou / len(val_list), t), file=mylog)
        print("train iou: %.6f" % (t_iou / len(train_list)), file=mylog)

        if loss < train_min_loss:
            train_min_loss = loss
            torch.save(net.state_dict(), os.path.join(args.save_path, 'train.pth'))
            print("INFO: save train model", file=mylog)

        # print('[epoch: %3d/%3d]' % (epoch + 1, args.epoch), 'Current lr: ', optimizer.state_dict()['param_groups'][0]['lr'])

        # if vloss < valid_min_loss:
        #     valid_min_loss = vloss
        #     torch.save(model.state_dict(), os.path.join(args.save_path, 'valid.pth'))
        #     print("INFO: save valid_min_loss model")

        if v_iou > valid_max_ghost:
            valid_max_ghost = v_iou
            torch.save(net.state_dict(), os.path.join(args.save_path, 'iou.pth'))
            print("INFO: save valid model", file=mylog)


if __name__ == '__main__':
    main()




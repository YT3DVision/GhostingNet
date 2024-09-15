import argparse
import os
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.autograd import Variable
from torchvision import transforms

from DANet import RGBD_sal

# from MacroNet.focalloader import Focal
from dataset.lfm import LFM
from utils import pytorch_ssim, pytorch_iou
from utils.Miou import iou_mean
import torch.nn.functional as functional


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def bce_ssim_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + ssim_out + iou_out

    return loss

def main():
    train_min_loss = float('inf')
    valid_min_loss = float('inf')
    valid_max_glass = float(0)
    valid_max_ghost = float(0)

    parser = argparse.ArgumentParser(description='LFSalient Training')

    # dataset settings
    parser.add_argument("--batchSize", type=int, default=24, help="Training batch size")
    parser.add_argument("--workers", type=int, default=16, help="DataLoader Num workers")

    # optimizer settings
    parser.add_argument("--lr", type=float, default=4e-5, help='learning rate')
    parser.add_argument("--lr_decay", type=float, default=0.9, help='Weight decay')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Learning rate decay steps')
    parser.add_argument('--momentum', default=0.9, type=float, help='Gradient clip')

    # training settings
    parser.add_argument("--epoch", type=int, default=400, help='')
    parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
    parser.add_argument("--use_GPU", type=bool, default=True, help='')
    parser.add_argument("--data_path", type=str, default="./LFM_V2/", help='Path to LFM')  # synthetic

    # parser.add_argument("--model_path", type=str, default="/data/gjhLF/logs/Focal_VGG_dac_filp_selayer/iou.pth", help='')
    # parser.add_argument("--sav e_path", type=str, default="/data/gjhLF/logs
    # /Focal_VGG_dac_filp_selayer200-400", help='')
    # parser.add_argument("--save_path", type=str, default="D:/.00_code/logs/cv_depth_res101_conv_1e-6-2", help='')
    parser.add_argument("--model_path", type=str, default="./compare/DANet ", help='')
    parser.add_argument("--save_path", type=str, default="./compare/DANet", help='')
    parser.add_argument("--log_path", type=str, default="./compare/DANet.log", help='')
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
    model = RGBD_sal()
    net = model.cuda()

    # loss
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_BCE = nn.BCELoss().cuda()
    criterion_MAE = nn.L1Loss().cuda()
    criterion_MSE = nn.MSELoss().cuda()

    # ------- 4. define optimizer --------
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    print("---Info: Start Training...")
    curr_iter = 0
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
            mask = Variable(mask)
            depth = Variable(depth)

            cv = cv.cuda()
            mask = mask.cuda()
            depth = depth.cuda()

            outputs,outputs_fg,outputs_bg,attention1,attention2,attention3,attention4,attention5 = net(cv,depth) #hed

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            labels1 = functional.interpolate(mask, size=16, mode='bilinear')
            labels2 = functional.interpolate(mask, size=32, mode='bilinear')
            labels3 = functional.interpolate(mask, size=64, mode='bilinear')
            labels4 = functional.interpolate(mask, size=128, mode='bilinear')
            loss1 = criterion_BCE(attention1, labels1)
            loss2 = criterion_BCE(attention2, labels2)
            loss3 = criterion_BCE(attention3, labels3)
            loss4 = criterion_BCE(attention4, labels4)
            loss5 = criterion_BCE(attention5, mask)
            loss6 = criterion(outputs_fg, mask)
            loss7 = criterion(outputs_bg, (1 - mask))
            loss8 = criterion(outputs, mask)
            total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
            total_loss.backward()
            optimizer.step()

            curr_iter += 1

            # print statistics
            running_loss += total_loss.item()

            # calculate iou
            pred = outputs
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
                mask = Variable(mask)
                depth = Variable(depth)

                cv = cv.cuda()
                mask = mask.cuda()
                depth = depth.cuda()

                # y zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(cv,depth)  # hed

                loss8 = criterion(outputs, mask)
                vtotal_loss = loss8

                running_val_loss += vtotal_loss.item()

                # calculate iou
                pred = outputs
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

        if total_loss < train_min_loss:
            train_min_loss = total_loss
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

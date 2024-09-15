import argparse
import torch.optim as optim
import torch.nn as nn
import torch
import time
import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from config import swin_B
from model.GED_wo_cascade import Net
# from utils.load_GDD import *
from utils.Miou import iou_mean
from utils.loss import bce_ssim_loss
from utils.data_loader import make_dataSet

import platform
from tqdm import tqdm
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, filename="log_ab_ged_wo_cascade_hh.txt", filemode="a",
                    format='%(asctime)s %(message)s')  # 全局配置
logging.info("Start train_hh ...")


def train():
    train_min_loss = float('inf')
    valid_max_glass = float(0)
    valid_max_ghost = float(0)

    scale = 384

    parser = argparse.ArgumentParser(description="PyTorch Glass Detection Example")
    parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=200, help='')
    parser.add_argument("--gpu_id", type=str, default="1", help='GPU id')
    parser.add_argument("--use_GPU", type=bool, default=True, help='')
    parser.add_argument("--data_path", type=str,
                        default=r"E:\PycharmProjects\GEGD2" if platform.system() == "Windows" else "/data/GEGD2",
                        help='')
    parser.add_argument("--save_path", type=str, default="./logs/table6_ged_wo_cascade_hh_5e-5", help='')
    parser.add_argument("--lr", type=float, default=5e-5, help='')
    opt = parser.parse_args()

    pz3 = torch.zeros(opt.batchSize, 1024, 12, 12).cuda()
    pz2 = torch.zeros(opt.batchSize, 512, 24, 24).cuda()
    pz1 = torch.zeros(opt.batchSize, 256, 48, 48).cuda()
    pz0 = torch.zeros(opt.batchSize, 128, 96, 96).cuda()

    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    rgb_transform = transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.ToTensor()
    ])

    grey_transform = transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.ToTensor()
    ])

    # load dataset
    print('INFO:Loading dataset ...\n')
    logging.info("INFO:Loading dataset ...")
    dataset_train = make_dataSet(opt.data_path, train=True, rgb_transform=rgb_transform, grey_transform=grey_transform)
    dataset_valid = make_dataSet(opt.data_path, train=False, rgb_transform=rgb_transform, grey_transform=grey_transform)

    loader_train = DataLoader(dataset=dataset_train, num_workers=2, batch_size=opt.batchSize, shuffle=True,
                              drop_last=True)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=2, batch_size=opt.batchSize, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    logging.info("# of training samples: %d\n" % int(len(dataset_train)))
    print("# of valid samples: %d\n" % int(len(dataset_valid)))
    logging.info("# of valid samples: %d\n" % int(len(dataset_valid)))

    # ##########create model###############
    model = Net(backbone_path=swin_B)
    if opt.use_GPU:
        model = model.cuda()

    # ##### optim #######
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # ##### Loss #######
    bce = nn.BCELoss()
    criterion = nn.L1Loss()
    MseCriterion = nn.MSELoss()

    def shift_loss(s0, gt, with_shift):
        # print(with_shift.numel())
        s0 = torch.sum(torch.mean(((s0 - gt) ** 2).view(with_shift.numel(), -1), 1) * with_shift)
        loss = s0
        return loss

    for epoch in range(opt.epochs):
        # ####### train ########
        start = time.time()
        model.train()
        model.zero_grad()

        glass_loss_sum = 0
        train_loss_sum = 0
        train_r1_sum = 0
        train_r2_sum = 0
        train_w_sum = 0
        train_h_sum = 0
        train_ghost_sum = 0
        train_shift_sum = 0
        train_glass_sum = 0
        train_per_sum = 0
        t_glass_iou = 0
        t_ghost_iou = 0

        for idx, (input_data, glass, ghost, r1, r2, h, w, with_shift) in enumerate(tqdm(loader_train)):
            input_data = Variable(input_data)
            ghost = Variable(ghost)
            glass = Variable(glass)
            r1 = Variable(r1)
            r2 = Variable(r2)
            h = Variable(h)
            w = Variable(w)
            with_shift = Variable(with_shift)

            if opt.use_GPU:
                input_data = input_data.cuda()
                ghost = ghost.cuda()
                glass = glass.cuda()
                r1 = r1.cuda()
                r2 = r2.cuda()
                h = h.cuda()
                w = w.cuda()
                with_shift = with_shift.cuda()

            optimizer.zero_grad()

            h0, w0, per0, r1_0, r2_0, g0 = model(input_data)

            lossH = shift_loss(h0, h, with_shift)

            lossW = shift_loss(w0, w, with_shift)

            lossp0 = MseCriterion(per0, pz0)

            lossPer = lossp0

            lossr10 = bce(r1_0, r1)

            lossR1 = lossr10

            lossr20 = bce(r2_0, r2)

            lossR2 = lossr20

            lossg0 = bce(g0, ghost)

            lossGhost = lossg0

            loss = lossR1 + lossR2 + 3 * lossGhost + 0.1 * lossPer + 0.1 * lossW + 0.1 * lossH
            loss.backward()

            train_loss_sum += loss.item()
            train_per_sum += lossPer.item()
            train_r1_sum += lossR1.item()
            train_r2_sum += lossR2.item()
            train_w_sum += lossW.item()
            train_h_sum += lossH.item()
            train_ghost_sum += lossGhost.item()
            optimizer.step()

        # ###### Validing ######

        model.eval()
        model.zero_grad()
        v_ghost_iou = 0
        v_glass_iou = 0
        valid_loss_sum = 0

        with torch.no_grad():
            for idx, (input_data, glass, ghost, r1, r2, h, w, with_shift) in enumerate(tqdm(loader_valid)):
                input_data = Variable(input_data)
                ghost = Variable(ghost)
                glass = Variable(glass)
                r1 = Variable(r1)
                r2 = Variable(r2)
                h = Variable(h)
                w = Variable(w)
                with_shift = Variable(with_shift)

                if opt.use_GPU:
                    input_data = input_data.cuda()
                    ghost = ghost.cuda()
                    glass = glass.cuda()
                    r1 = r1.cuda()
                    r2 = r2.cuda()
                    h = h.cuda()
                    w = w.cuda()
                    with_shift = with_shift.cuda()

                optimizer.zero_grad()

                h0, w0, per0, r1_0, r2_0, g0 = model(input_data)

                pred = g0
                label = ghost
                bs, _, _, _ = label.shape

                temp1 = pred.data.squeeze(1)
                temp2 = label.data.squeeze(1)
                for i in range(bs):
                    a = temp1[i, :, :]
                    b = temp2[i, :, :]
                    a = torch.round(a).squeeze(0).int().detach().cpu()
                    b = torch.round(b).squeeze(0).int().detach().cpu()
                    v_ghost_iou += iou_mean(a, b, 1)

                torch.cuda.empty_cache()

        end = time.time()
        t = end - start

        print('INFO: epoch:{},tl:{},tpl:{},twl:{},thl:{},tr1:{},tr2:{},tgs:{},V_gh:{}, time:{}'.format(
            epoch + 1,
            round(train_loss_sum / len(
                loader_train), 6),
            round(train_per_sum / len(
                loader_train), 6),
            round(train_w_sum / len(
                loader_train), 6),
            round(train_h_sum / len(
                loader_train), 6),
            round(train_r1_sum / len(
                loader_train), 6),
            round(train_r2_sum / len(
                loader_train), 6),
            round(train_ghost_sum / len(
                loader_train), 6),
            round(v_ghost_iou / len(
                loader_valid) / opt.batchSize * 100, 2),
            round(t, 2)))

        logging.info('INFO: epoch:{},tl:{},tpl:{},twl:{},thl:{},tr1:{},tr2:{},tgs:{},V_gh:{}, time:{}'.format(
            epoch + 1,
            round(train_loss_sum / len(
                loader_train), 6),
            round(train_per_sum / len(
                loader_train), 6),
            round(train_w_sum / len(
                loader_train), 6),
            round(train_h_sum / len(
                loader_train), 6),
            round(train_r1_sum / len(
                loader_train), 6),
            round(train_r2_sum / len(
                loader_train), 6),
            round(train_ghost_sum / len(
                loader_train), 6),
            round(v_ghost_iou / len(
                loader_valid) / opt.batchSize * 100, 2),
            round(t, 2)))

        # del temporary outputs and loss
        # del final_glass, lossGlass, loss1, loss2, loss3, loss0, lossfuse, lossfinal

        if train_loss_sum < train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'train_min.pth'))
            print("INFO: save train_min model")
            logging.info("INFO: save train_min model")

        if v_ghost_iou > valid_max_ghost:
            valid_max_ghost = v_ghost_iou
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'ghost_max.pth'))
            print("INFO: save ghost model")
            logging.info("INFO: save ghost model")

    #    if epoch % 10 == 0 :
    #        name = 'epoch' + str(epoch) + '.pth'
    #        torch.save(model.state_dict(), os.path.join(opt.save_path, name))
    #        print("INFO: save model")


if __name__ == '__main__':
    train()

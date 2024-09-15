import argparse
import torch.optim as optim
import torch.nn as nn
import torch
import time
import os
import logging
from torchvision import transforms
from torch.utils.data import DataLoader

from config import swin_B
from model.net2_z2 import Net

from utils.Miou import iou_mean
from utils.loss import bce_ssim_loss
from utils.data_loader import make_dataSet

import warnings
from tqdm import tqdm
# from nb_log import get_logger
from datetime import datetime
import platform


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, filename="./log.txt", filemode="a", format='%(asctime)s %(message)s')  # 全局配置
# logger = get_logger("train-normal-info", is_add_stream_handler=True, log_filename="./train_hh.log")
logging.info("Start train_hh ...")


def get_transform(img_type):
    assert img_type in ["rgb", "grey"], "image type must be in ['rgb', 'grey']"
    rgb_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])

    grey_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    return rgb_transform if img_type == "rgb" else grey_transform


# ##### Loss #######
bce = nn.BCELoss()
criterion = nn.L1Loss()
MseCriterion = nn.MSELoss()


def main(opt):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    logging.info("Loading dataset ...")
    dataset_train = make_dataSet(opt.data_path, train=True, rgb_transform=get_transform("rgb"), grey_transform=get_transform("grey"))
    dataset_valid = make_dataSet(opt.data_path, train=False, rgb_transform=get_transform("rgb"), grey_transform=get_transform("grey"))

    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True, drop_last=True)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batchSize, shuffle=False)
    logging.info(f"training samples:{int(len(dataset_train))}")
    logging.info(f"valid samples:{int(len(dataset_valid))}")

    model = Net(backbone_path=swin_B).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    if opt.resume:
        logging.info(f"Resuming training, loading {opt.resume}...")
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        opt.start_epoch = checkpoint['epoch'] + 1

    train_min_loss = float('inf')
    valid_max_glass = float(0)
    valid_max_ghost = float(0)

    pz3 = torch.zeros(opt.batchSize, 1024, 12, 12).cuda()
    pz2 = torch.zeros(opt.batchSize, 512, 24, 24).cuda()
    pz1 = torch.zeros(opt.batchSize, 256, 48, 48).cuda()
    pz0 = torch.zeros(opt.batchSize, 128, 96, 96).cuda()

    for epoch in range(opt.start_epoch, opt.epochs):
        start = time.time()
        model.train()
        model.zero_grad()

        train_loss_sum = 0
        train_r1_sum = 0
        train_r2_sum = 0
        train_w_sum = 0
        train_h_sum = 0
        train_ghost_sum = 0
        train_glass_sum = 0
        train_per_sum = 0

        pbar = tqdm(loader_train)
        for idx, (rgb_image, glass, ghost, r1, r2, h, w, is_with_shift) in enumerate(pbar):
            pbar.set_description(f"train-epoch: {epoch}")
            rgb_image = rgb_image.to(device)
            ghost = ghost.to(device)
            glass = glass.to(device)
            r1 = r1.to(device)
            r2 = r2.to(device)
            h = h.to(device)
            w = w.to(device)
            is_with_shift = is_with_shift.to(device)

            optimizer.zero_grad()

            h3, h2, h1, h0, \
            w3, w2, w1, w0, \
            per3, per2, per1, per0, \
            r1_3, r1_2, r1_1, r1_0, \
            r2_3, r2_2, r2_1, r2_0, \
            g3, g2, g1, g0, fg, \
            l3, l2, l1, l0, final_glass, ged_swin_encode_feature_fusion = model(rgb_image)
            lossH = shift_loss(h3, h2, h1, h0, h, is_with_shift)

            lossW = shift_loss(w3, w2, w1, w0, w, is_with_shift)

            lossp3 = MseCriterion(per3, pz3)
            lossp2 = MseCriterion(per2, pz2)
            lossp1 = MseCriterion(per1, pz1)
            lossp0 = MseCriterion(per0, pz0)

            lossPer = lossp3 + lossp2 + lossp1 + lossp0

            lossr13 = bce(r1_3, r1)
            lossr12 = bce(r1_2, r1)
            lossr11 = bce(r1_1, r1)
            lossr10 = bce(r1_0, r1)

            lossR1 = lossr13 + lossr12 + lossr11 + lossr10

            lossr23 = bce(r2_3, r2)
            lossr22 = bce(r2_2, r2)
            lossr21 = bce(r2_1, r2)
            lossr20 = bce(r2_0, r2)

            lossR2 = lossr23 + lossr22 + lossr21 + lossr20

            lossg3 = bce(g3, ghost)
            lossg2 = bce(g2, ghost)
            lossg1 = bce(g1, ghost)
            lossg0 = bce(g0, ghost)
            lossgfuse = bce(fg, ghost)

            lossGhost = lossgfuse + lossg3 + lossg2 + lossg1 + lossg0

            loss3 = bce_ssim_loss(l3, glass)
            loss2 = bce_ssim_loss(l2, glass)
            loss1 = bce_ssim_loss(l1, glass)
            loss0 = bce_ssim_loss(l0, glass)
            lossfinal = bce_ssim_loss(final_glass, glass)

            lossGlass = lossfinal + loss3 + loss2 + loss1 + loss0

            loss = lossR1 + lossR2 + 2 * lossGlass + 4 * lossGhost + 0.1 * lossPer + 0.1 * lossW + 0.1 * lossH

            loss.backward()

            train_loss_sum += loss.item()
            train_per_sum += lossPer.item()
            train_r1_sum += lossR1.item()
            train_r2_sum += lossR2.item()
            train_w_sum += lossW.item()
            train_h_sum += lossH.item()
            train_ghost_sum += lossGhost.item()
            train_glass_sum += lossGlass.item()
            optimizer.step()
            # pbar.set_description(f"train-epoch: {epoch}")


        ####### Validing ######
        model.eval()
        model.zero_grad()
        v_ghost_iou = 0
        v_glass_iou = 0

        with torch.no_grad():
            pbar_valid = tqdm(loader_valid)
            for idx, (rgb_image, glass, ghost, r1, r2, h, w, is_with_shift) in enumerate(pbar_valid):
                pbar_valid.set_description(f"valid-epoch: {epoch}")
                rgb_image = rgb_image.to(device)
                ghost = ghost.to(device)
                glass = glass.to(device)

                optimizer.zero_grad()

                h3, h2, h1, h0, \
                w3, w2, w1, w0, \
                per3, per2, per1, per0, \
                r1_3, r1_2, r1_1, r1_0, \
                r2_3, r2_2, r2_1, r2_0, \
                g3, g2, g1, g0, fg, \
                l3, l2, l1, l0, final_glass, ged_swin_encode_feature_fusion = model(rgb_image)

                pred = fg
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

                pred = final_glass
                label = glass
                bs, _, _, _ = label.shape

                temp1 = pred.data.squeeze(1)
                temp2 = label.data.squeeze(1)
                for i in range(bs):
                    a = temp1[i, :, :]
                    b = temp2[i, :, :]
                    a = torch.round(a).squeeze(0).int().detach().cpu()
                    b = torch.round(b).squeeze(0).int().detach().cpu()
                    v_glass_iou += iou_mean(a, b, 1)

                torch.cuda.empty_cache()

        end = time.time()
        t = end - start

        logging.info('INFO: epoch:{},tl:{},tpl:{},twl:{},thl:{},tr1:{},tr2:{},tgs:{},tgl:{},V_gh:{},V_gl:{}, time:{}'.format(
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
            round(train_glass_sum / len(
                loader_train), 6),
            round(v_ghost_iou / len(
                loader_valid) / opt.batchSize * 100, 2),
            round(v_glass_iou / len(
                loader_valid) / opt.batchSize * 100, 2),
            round(t, 2)))

        print('INFO: epoch:{},tl:{},tpl:{},twl:{},thl:{},tr1:{},tr2:{},tgs:{},tgl:{},V_gh:{},V_gl:{}, time:{}'.format(
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
            round(train_glass_sum / len(
                loader_train), 6),
            round(v_ghost_iou / len(
                loader_valid) / opt.batchSize * 100, 2),
            round(v_glass_iou / len(
                loader_valid) / opt.batchSize * 100, 2),
            round(t, 2)))

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if train_loss_sum < train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(save_file, os.path.join(opt.save_path, 'train_min.pth'))
            print("INFO: save train_min model")

        if v_glass_iou > valid_max_glass:
            valid_max_glass = v_glass_iou
            torch.save(save_file, os.path.join(opt.save_path, 'glass_max.pth'))
            print("INFO: save glass model")

        if v_ghost_iou > valid_max_ghost:
            valid_max_ghost = v_ghost_iou
            torch.save(save_file, os.path.join(opt.save_path, 'ghost_max.pth'))
            print("INFO: save ghost model")


def shift_loss(s3, s2, s1, s0, gt, is_with_shift):
    s3 = torch.sum(torch.mean(((s3 - gt) ** 2).view(is_with_shift.numel(), -1), 1) * is_with_shift)
    s2 = torch.sum(torch.mean(((s2 - gt) ** 2).view(is_with_shift.numel(), -1), 1) * is_with_shift)
    s1 = torch.sum(torch.mean(((s1 - gt) ** 2).view(is_with_shift.numel(), -1), 1) * is_with_shift)
    s0 = torch.sum(torch.mean(((s0 - gt) ** 2).view(is_with_shift.numel(), -1), 1) * is_with_shift)
    loss = s3 + s2 + s1 + s0
    return loss


def parse_args():
    parser = argparse.ArgumentParser("Pytorch GhostNet Training")
    parser.add_argument("--batchSize", type=int, default=2, help="Training batch size")
    parser.add_argument("--start-epoch", type=int, default=0, help='')
    parser.add_argument("--epochs", type=int, default=300, help='')
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--data_path", type=str, default=r"E:\PycharmProjects\GhosetNetV3\GEGD" if platform.system() == "Windows" else r"/data/dataSet/GEGD", help='')
    parser.add_argument("--save_path", type=str, default="", help='')  # 88.95
    parser.add_argument("--lr", type=float, default=1e-5, help='')
    parser.add_argument('--resume', default="", help='resume from checkpoint')
    return parser.parse_args()


def make_dir(args):
    if os.path.exists(args.save_path) is False:
        results_file = "model-{}-".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
        default_path = os.path.join("./logs", results_file + Net()._get_name())
        os.makedirs(default_path)
        args.save_path = default_path


if __name__ == '__main__':
    args = parse_args()
    make_dir(args)
    main(args)

import argparse
import torch.optim as optim
import torch
import time
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from model.net2_z2_GSD import Net as GSDNet
from model.net2_z2 import Net as GEDNet
from config import *
from utils.load_GDD_net2_z2 import *
from utils.Miou import iou_mean
from utils.loss import *
import platform
from tqdm import tqdm

scale = 384

parser = argparse.ArgumentParser(description="PyTorch Glass Detection Example")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--epochs", type=int, default=400, help='')
parser.add_argument("--device", default="cuda", help="training device")
parser.add_argument("--data_path", type=str,
                    default=r"E:\PycharmProjects\GDD" if platform.system() == "Windows" else "/data/GDD", help='')
parser.add_argument("--ged_model_path", type=str, default="./logs/model-20230719-133613-net2_z2/ghost_max.pth", help='')
parser.add_argument("--save_path", type=str, default="./logs/GSDNet/net2_z2_GDD", help='')
parser.add_argument("--lr", type=float, default=1e-5, help='')

opt = parser.parse_args()

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

img_transform = transforms.Compose([
    transforms.Resize((scale, scale)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    transforms.ToTensor()
])

# ##### Loss #######
criterion = nn.BCELoss()
MseCriterion = nn.MSELoss()

device = torch.device(opt.device if torch.cuda.is_available() else "cpu")


def infer():
    """
    由GED子网络生成ghost和ged_fusion
    对train和test都有生成
    """
    # 1 在GEGD上训练好的模型，用于生成ghost和g_fusion
    ged_model = GEDNet(backbone_path=swin_B).to(device)
    ged_model.load_state_dict(torch.load(opt.ged_model_path)["model"])
    ged_model.eval()

    # 2 check
    train_data_path = os.path.join(opt.data_path, "train")
    test_data_path = os.path.join(opt.data_path, "test")
    num_img = len(os.listdir(os.path.join(train_data_path, "image")))
    ghost_npy_train_path = os.path.join(train_data_path, "ghostnpy")
    if os.path.exists(ghost_npy_train_path) and len(os.listdir(ghost_npy_train_path)) == num_img:
        return

    # 3 generate
    ghost_npy_test_path = os.path.join(test_data_path, "ghostnpy")
    os.makedirs(ghost_npy_train_path)
    os.makedirs(ghost_npy_test_path)

    gedfusion_npy_train_path = os.path.join(train_data_path, "gedfusionnpy")
    gedfusion_npy_test_path = os.path.join(test_data_path, "gedfusionnpy")
    os.makedirs(gedfusion_npy_train_path)
    os.makedirs(gedfusion_npy_test_path)

    with torch.no_grad():
        generate_npy(ghost_npy_train_path, gedfusion_npy_train_path, ged_model)
        generate_npy(ghost_npy_test_path, gedfusion_npy_test_path, ged_model)


def generate_npy(ghost_path, gedfusion_path, ged_model):
    var = "train" if "train" in ghost_path else "test"
    img_list = [img_name for img_name in os.listdir(os.path.join(opt.data_path, var, "image"))]
    for idx, (img_name) in enumerate(img_list):
        img = Image.open(os.path.join(opt.data_path, var, "image", img_name))
        img = img_transform(img).unsqueeze(0).to(torch.device(device))
        result_array = ged_model(img)
        ged_swin_encode_feature_fusion = result_array[-1]
        ghost = result_array[24]
        np.save(os.path.join(ghost_path, img_name[:-4] + ".npy"), ghost.squeeze(0).squeeze(0).cpu())
        np.save(os.path.join(gedfusion_path, img_name[:-4] + ".npy"),
                ged_swin_encode_feature_fusion.squeeze(0).squeeze(0).cpu())


def train():
    gsd_model = GSDNet(backbone_path=swin_B).to(device)

    optimizer = optim.Adam(gsd_model.parameters(), lr=opt.lr)

    # load dataset
    print('INFO:Loading dataset ...\n')
    dataset_train = make_dataSet(opt.data_path, train=True, rgb_transform=rgb_transform, grey_transform=grey_transform)
    dataset_valid = make_dataSet(opt.data_path, train=False, rgb_transform=rgb_transform, grey_transform=grey_transform)

    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batchSize, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    print("# of valid samples: %d\n" % int(len(dataset_valid)))

    train_min_loss = float('inf')
    valid_max_glass = float(0)
    for epoch in range(opt.epochs):
        # ####### train ########
        start = time.time()
        gsd_model.train()
        gsd_model.zero_grad()

        glass_loss_sum = 0
        train_loss_sum = 0

        for idx, (image, glass, ghost, ged_swin_encode_feature_fusion) in enumerate(tqdm(loader_train)):
            image = image.to(device)
            glass = glass.to(device)
            ghost = ghost.to(device)
            ged_swin_encode_feature_fusion = ged_swin_encode_feature_fusion.to(device)

            optimizer.zero_grad()

            l3, l2, l1, l0, final_glass = gsd_model(image, ghost, ged_swin_encode_feature_fusion)

            loss3 = bce_ssim_loss(l3, glass)
            loss2 = bce_ssim_loss(l2, glass)
            loss1 = bce_ssim_loss(l1, glass)
            loss0 = bce_ssim_loss(l0, glass)
            lossfinal = bce_ssim_loss(final_glass, glass)

            lossGlass = 2 * lossfinal + loss3 + loss2 + loss1 + loss0

            loss = lossGlass
            loss.backward()

            train_loss_sum += lossGlass.item()
            glass_loss_sum += lossGlass.item()
            optimizer.step()

        # ###### Validing ######

        gsd_model.eval()
        gsd_model.zero_grad()
        v_glass_iou = 0

        with torch.no_grad():
            for idx, (image, glass, ghost, ged_swin_encode_feature_fusion) in enumerate(tqdm(loader_valid)):
                image = image.to(device)
                glass = glass.to(device)
                ghost = ghost.to(device)
                ged_swin_encode_feature_fusion = ged_swin_encode_feature_fusion.to(device)

                l3, l2, l1, l0, final_glass = gsd_model(image, ghost, ged_swin_encode_feature_fusion)

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

        print('INFO: epoch:{},tl:{},gl_l:{},V_gl:{}, time:{}'.format(epoch + 1,
                                                                     round(train_loss_sum / len(
                                                                         loader_train), 6),
                                                                     round(glass_loss_sum / len(
                                                                         loader_train), 6),
                                                                     round(v_glass_iou / len(
                                                                         loader_valid) / opt.batchSize * 100, 2),
                                                                     round(t, 2)))

        if train_loss_sum < train_min_loss:
            train_min_loss = train_loss_sum
            torch.save(gsd_model.state_dict(), os.path.join(opt.save_path, 'train_min.pth'))
            print("INFO: save train_min model")

        if v_glass_iou > valid_max_glass:
            valid_max_glass = v_glass_iou
            torch.save(gsd_model.state_dict(), os.path.join(opt.save_path, 'glass_max.pth'))
            print("INFO: save glass model")


if __name__ == '__main__':
    infer()
    # train()

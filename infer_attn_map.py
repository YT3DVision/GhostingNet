from visualizer import get_local
get_local.activate()
import argparse

import os
import time

from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

from config import swin_B
from model.net2 import Net
from tqdm import tqdm
import matplotlib.pyplot as plt


scale = 384

# parser
parser = argparse.ArgumentParser(description="PyTorch Mirror Detection Example")
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--use_GPU", type=bool, default=True, help='')
# parser.add_argument("--data_path", type=str, default=r"E:\PycharmProjects\GhosetNetV3\GEGD\test\image", help='')
parser.add_argument("--data_path", type=str, default=r"E:\PycharmProjects\GhosetNetV3\GEGD\train\image", help='')
# parser.add_argument("--model_path", type=str, default="./logs/Ver.0.3.30.s2_2", help='')  #  now sota
# parser.add_argument("--result_path", type=str, default="./result/Ver.0.3.30.s2_2_Ours", help='')
# parser.add_argument("--model_path", type=str, default="./logs/V2023629_net_z/glass_max.pth", help='')  #  now sota
parser.add_argument("--model_path", type=str, default="./logs/Ver.4.28.net2/best.pth", help='')  # now sota
# parser.add_argument("--model_path", type=str, default="./logs/Ver.5.1.net2/58.32_95.56.pth", help='')  #  5.3 this paper in net2
# parser.add_argument("--model_path", type=str, default="./logs/Ver.4.28.net2/train_min.pth", help='')  #  now sota

# parser.add_argument("--result_path", type=str, default="./result/Ours_5_3_train_min.3", help='') # Ours final model
# parser.add_argument("--result_path", type=str, default="./result/Ours_5_3_train_min.3", help='')
parser.add_argument("--result_path", type=str, default="./Run_debug", help='')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

to_pil = transforms.ToPILImage()

img_path = os.path.join(opt.result_path, 'image')
r1_path = os.path.join(opt.result_path, 'r1')
r2_path = os.path.join(opt.result_path, 'r2')
ghost_path = os.path.join(opt.result_path, 'ghost')
glass_path = os.path.join(opt.result_path, 'glass')
h_path = os.path.join(opt.result_path, 'h_mask')
w_path = os.path.join(opt.result_path, 'w_mask')

if not os.path.isdir(opt.result_path):
    os.makedirs(opt.result_path)
if not os.path.isdir(img_path):
    os.makedirs(img_path)
if not os.path.isdir(r1_path):
    os.makedirs(r1_path)
if not os.path.isdir(r2_path):
    os.makedirs(r2_path)
if not os.path.isdir(ghost_path):
    os.makedirs(ghost_path)
if not os.path.isdir(glass_path):
    os.makedirs(glass_path)
if not os.path.isdir(h_path):
    os.makedirs(h_path)
if not os.path.isdir(w_path):
    os.makedirs(w_path)

img_transform = transforms.Compose([
    transforms.Resize((scale, scale)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    transforms.ToTensor()
])

# ##########create model###############
model = Net(backbone_path=swin_B)
if opt.use_GPU:
    model = model.cuda()


def _test():
    state_dict = torch.load(opt.model_path)
    model.load_state_dict(state_dict, strict=False)

    # model.eval()
    with torch.no_grad():
        start = time.time()
        img_list = [img_name for img_name in os.listdir(os.path.join(opt.data_path))]
        # img_list = ["82.jpg"]
        print(img_list)

        for idx, img_name in enumerate(tqdm(img_list)):
            img = Image.open(os.path.join(opt.data_path, img_name))

            w, h = img.size

            img_var = Variable(img_transform(img).unsqueeze(0)).cuda()

            h3, h2, h1, h0, \
            w3, w2, w1, w0, \
            per3, per2, per1, per0, \
            r1_3, r1_2, r1_1, r1_0, \
            r2_3, r2_2, r2_1, r2_0, \
            g3, g2, g1, g0, fg, \
            l3, l2, l1, l0, final_glass = model(img_var)
            cache = get_local.cache
            # print(list(cache.keys()))
            features = cache['Net.forward']
            feature = features[-1]
            S1F1 = feature[0]
            S1F2 = feature[1]
            S1F3 = feature[2]
            S1F4 = feature[3]
            S2F1 = feature[4]
            S2F2 = feature[5]
            S2F3 = feature[6]
            S2F4 = feature[7]
            plt.imsave('./images/' + img_name[:-4] + '_S1F1.png', S1F1)
            plt.imsave('./images/' + img_name[:-4] + '_S1F2.png', S1F2)
            plt.imsave('./images/' + img_name[:-4] + '_S1F3.png', S1F3)
            plt.imsave('./images/' + img_name[:-4] + '_S1F4.png', S1F4)
            plt.imsave('./images/' + img_name[:-4] + '_S2F1.png', S2F1)
            plt.imsave('./images/' + img_name[:-4] + '_S2F2.png', S2F2)
            plt.imsave('./images/' + img_name[:-4] + '_S2F3.png', S2F3)
            plt.imsave('./images/' + img_name[:-4] + '_S2F4.png', S2F4)
            # cache.clear()
    # attention = attentions[-2]
    # attention = attention.squeeze()
    # attention = attention.mean(axis=0)
    # attention = attention[0]
    # attention = attention.reshape(12, 12)
    # plt.imshow(attention)
    # plt.show()
    # attention = attentions[20]
    # attention = attention.squeeze()
    # attention = attention.mean(axis=0)
    # attention = attention[0]
    # attention = attention.reshape(12, 12)
    # plt.imshow(attention)
    # plt.show()


            # out_r1 = r1_0.squeeze(0).cpu()
            # out_r2 = r2_0.squeeze(0).cpu()
            # out_ghost = fg.squeeze(0).cpu()
            # out_glass = final_glass.squeeze(0).cpu()
            #
            # out_r1 = np.array(transforms.Resize((h, w))(to_pil(out_r1)))
            # Image.fromarray(out_r1).save(os.path.join(r1_path, img_name[:-4] + ".png"))
            #
            # out_r2 = np.array(transforms.Resize((h, w))(to_pil(out_r2)))
            # Image.fromarray(out_r2).save(os.path.join(r2_path, img_name[:-4] + ".png"))
            #
            # out_glass = np.array(transforms.Resize((h, w))(to_pil(out_glass)))
            # Image.fromarray(out_glass).save(os.path.join(glass_path, img_name[:-4] + ".png"))
            #
            # # out_r1 = r1_0.squeeze(0).cpu()
            # # out_r2 = r2_0.squeeze(0).cpu()
            # out_ghost = out_ghost.squeeze(0).cpu()
            #
            # # np.save(os.path.join(r1_path, img_name[:-4] + ".npy"), out_r1)
            # # np.save(os.path.join(r2_path, img_name[:-4] + ".npy"), out_r2)
            # np.save(os.path.join(ghost_path, img_name[:-4] + ".npy"), out_ghost)
            #
            # out_ghost = np.array(transforms.Resize((h, w))(to_pil(out_ghost)))
            # Image.fromarray(out_ghost).save(os.path.join(ghost_path, img_name[:-4] + ".png"))
            #
            # out_h = torch.nn.functional.sigmoid(h0).cpu()
            # out_w = torch.nn.functional.sigmoid(w0).cpu()
            # out_h = out_h.squeeze(0)
            # out_w = out_w.squeeze(0)
            #
            # img_h = np.array(transforms.Resize((h, w))(to_pil(out_h)))
            # img_w = np.array(transforms.Resize((h, w))(to_pil(out_w)))
            # Image.fromarray(img_h).save(os.path.join(h_path, img_name[:-4] + ".png"))
            # Image.fromarray(img_w).save(os.path.join(w_path, img_name[:-4] + ".png"))
            #
            # np.save(os.path.join(h_path, img_name[:-4] + ".npy"), out_h)
            # np.save(os.path.join(w_path, img_name[:-4] + ".npy"), out_w)


if __name__ == '__main__':
    _test()

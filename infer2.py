import argparse

import os
import time

from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np

from model_shift2 import *
scale = 384

# parser
parser = argparse.ArgumentParser(description="PyTorch Mirror Detection Example")
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--use_GPU", type=bool, default=True, help='')
# parser.add_argument("--data_path", type=str, default="/data/dataSet/mix", help='')
# parser.add_argument("--data_path", type=str, default="/data/gjh2/GSD/train/image", help='')
# parser.add_argument("--data_path", type=str, default="/data/GDD/test/image", help='')
parser.add_argument("--data_path", type=str, default="/data/dataSet/GEGD2/test/image", help='')
# parser.add_argument("--model_path", type=str, default="./logs/Ver.0.3.30.s2_2", help='')  #  now sota
# parser.add_argument("--result_path", type=str, default="./result/Ver.0.3.30.s2_2_Ours", help='')
parser.add_argument("--model_path", type=str, default="./logs/Ver4.22_shift2-Here/Ghost.pth", help='')  #  now sota
parser.add_argument("--result_path", type=str, default="./result/GEGD_test", help='')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

to_pil = transforms.ToPILImage()

img_path = os.path.join(opt.result_path, 'image')
r1_path = os.path.join(opt.result_path, 'r1')
r2_path = os.path.join(opt.result_path, 'r2')
ghost_path = os.path.join(opt.result_path, 'ghost')
ghostnpy_path = os.path.join(opt.result_path, 'ghostnpy')
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
if not os.path.isdir(ghostnpy_path):
    os.makedirs(ghostnpy_path)
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
model = Net(backbone_path=None)
if opt.use_GPU:
    model = model.cuda()

def test():
    state_dict = torch.load(opt.model_path)
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    with torch.no_grad():
        start = time.time()
        img_list = [img_name for img_name in os.listdir(os.path.join(opt.data_path))]
        print(img_list)

        for idx, img_name in enumerate(img_list):
            img = Image.open(os.path.join(opt.data_path, img_name))

            w, h = img.size

            img_var = Variable(img_transform(img).unsqueeze(0)).cuda()

            h3, h2, h1, h0, \
            w3, w2, w1, w0, \
            p3, p2, p1, p0, \
            r1_3, r1_2, r1_1, r1_0, \
            r2_3, r2_2, r2_1, r2_0, \
            g3, g2, g1, g0, fg = model(img_var)

            out_r1 = r1_3.squeeze(0).cpu()
            out_r2 = r2_3.squeeze(0).cpu()
            out_ghost = fg.squeeze(0).cpu()

            out_r1 = np.array(transforms.Resize((h, w))(to_pil(out_r1)))
            Image.fromarray(out_r1).save(os.path.join(r1_path, img_name[:-4] + ".png"))

            out_r2 = np.array(transforms.Resize((h, w))(to_pil(out_r2)))
            Image.fromarray(out_r2).save(os.path.join(r2_path, img_name[:-4] + ".png"))

            out_ghost = np.array(transforms.Resize((h, w))(to_pil(out_ghost)))
            Image.fromarray(out_ghost).save(os.path.join(ghost_path, img_name[:-4] + ".png"))
            
            out_h = h0.cpu()
            out_w = w0.cpu()
            out_ghost = fg.cpu()
            out_ghost = fg.squeeze(0)
            out_ghost = out_ghost.cpu()
            out_h = out_h.squeeze(0)
            out_w = out_w.squeeze(0)
            

            np.save(os.path.join(h_path, img_name[:-4] + ".npy"), out_h)
            np.save(os.path.join(w_path, img_name[:-4] + ".npy"), out_w)
            np.save(os.path.join(ghostnpy_path, img_name[:-4] + ".npy"), out_ghost)

if __name__ == '__main__':
    test()

import argparse

import os
import time
import torch

from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from PIL import Image
import numpy as np

# from sem import SEM_Net
scale = 384

# parser
parser = argparse.ArgumentParser(description="PyTorch Mirror Detection Example")
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--use_GPU", type=bool, default=True, help='')
# parser.add_argument("--data_path", type=str, default="/data/dataSet/GEGD/test/image", help='')
parser.add_argument("--data_path", type=str, default="/data/Nips/Ver1.1/result/Ver0.4.10.s1.mix/", help='')
# parser.add_argument("--model_path", type=str, default="./logs/Ver.0.3.30.s2_2", help='')  #  now sota
# parser.add_argument("--result_path", type=str, default="./result/Ver.0.3.30.s2_2_Ours", help='')
# parser.add_argument("--model_path", type=str, default="./logs/Ver0.4.2.s1", help='')  #  now sota
parser.add_argument("--result_path", type=str, default="./result/sem.syn.mix", help='')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

to_pil = transforms.ToPILImage()

h_path = os.path.join(opt.result_path, 'h_mask')
w_path = os.path.join(opt.result_path, 'w_mask')

if not os.path.isdir(opt.result_path):
    os.makedirs(opt.result_path)
    os.makedirs(h_path)
if not os.path.isdir(w_path):
    os.makedirs(w_path)
if not os.path.isdir(h_path):
    os.makedirs(h_path)
    

img_transform = transforms.Compose([
    transforms.Resize((scale, scale)),
    transforms.ToTensor()
])

# ##########create model###############
model = SEM_Net()
if opt.use_GPU:
    model = model.cuda()



def _test():
    # model.load_state_dict(torch.load('/data/Nips/Ver1.1/logs/Ver4.7/glass_max.pth'))
    new_state_dict = OrderedDict()
    state_dict2 = torch.load('/data/Nips/Ver1.1/logs/Ver0.4.10.s1/ghost_max.pth')
    for k, v in state_dict2.items():
        name = k[0:4]
        if name == 'SEM0':
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
        
    '''
    state_dict = torch.load('/data/Nips/Ver1.1/logs/Ver0.4.7.s1/ghost_max.pth')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        print(k)
        print('--------')
        name = k[0:3]
        if name == 'SEM':
            new_state_dict[k] = v
    state_dict2 = torch.load('/data/Nips/Ver1.1/logs/Ver4.7/glass_max.pth')
    for k, v in state_dict2.items():
        name = k[0:3]
        if name != 'SEM':
            new_state_dict[k] = v
    print("---start load pretrained modle of swin encoder---")
    for k, v in new_state_dict.items():
        print(k)
        print(v)
    '''
    model.load_state_dict(new_state_dict, strict=False)
    
    
    model.eval()
    with torch.no_grad():
        start = time.time()
        img_list = [img_name for img_name in os.listdir(os.path.join(opt.data_path, 'r1'))]
        print(img_list)

        for idx, img_name in enumerate(img_list):
            r1 = Image.open(os.path.join(opt.data_path, 'r1', img_name))
            r2 = Image.open(os.path.join(opt.data_path, 'r2', img_name))

            w, h = r1.size

            r1_var = Variable(img_transform(r1).unsqueeze(0)).cuda()
            r2_var = Variable(img_transform(r2).unsqueeze(0)).cuda()

            h0, w0 = model(r1_var, r2_var)
            
            out_h = h0.cpu()
            out_w = w0.cpu()
            out_h = out_h.squeeze(0)
            out_w = out_w.squeeze(0)

            np.save(os.path.join(h_path, img_name[:-4] + ".npy"), out_h)
            np.save(os.path.join(w_path, img_name[:-4] + ".npy"), out_w)

if __name__ == '__main__':
    _test()

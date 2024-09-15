import argparse
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import time
import os
from PIL import Image

# from model.GPGN_baseline import *
from model.GSDNet2 import *
from config import *
from utils.Miou import *
scale = 384

#
parser = argparse.ArgumentParser(description="PyTorch Mirror Detection Example")
parser.add_argument("--device", default="cuda", help="")
# parser.add_argument("--data_path", type=str, default="/data/GDD/test", help='')
parser.add_argument("--data_path", type=str, default=r"E:\PycharmProjects\GDD\test", help='')
# parser.add_argument("--data_path", type=str, default="/data/dataSet/GEGD_Real/test", help='')
# parser.add_argument("--save_path", type=str, default="./logs/GSDNet/GDD0501", help='') # 1104
parser.add_argument("--save_path", type=str, default="./logs/GSDNet/GDD_hh", help='')
# parser.add_argument("--result_path", type=str, default="./results/GDD_0501", help='')  # 1104
parser.add_argument("--result_path", type=str, default="./results/GDD_hh", help='')
# parser.add_argument('--useBN', action='store_true', help='enalbes batch normalization')

opt = parser.parse_args()
# print(opt.gpu_id[1])

img_transform = transforms.Compose([
    transforms.Resize((scale, scale)),
    transforms.ToTensor()
])

target_transform = transforms.Compose([
    transforms.Resize((scale, scale)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

glass_path = os.path.join(opt.result_path, 'glass')

if not os.path.isdir(opt.result_path):
    os.makedirs(opt.result_path)
if not os.path.isdir(glass_path):
    os.makedirs(glass_path)


def main():
    # ######## create Model #############
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = GSDNet(backbone_path=swin_B).to(device)

    model.load_state_dict(torch.load(os.path.join(opt.save_path, 'glass_max.pth')))

    model.eval()
    with torch.no_grad():

        start = time.time()
        img_list = [img_name for img_name in os.listdir(os.path.join(opt.data_path, 'image'))]
        print(img_list)

        for idx, img_name in enumerate(img_list):
            img = Image.open(os.path.join(opt.data_path, 'image', img_name))
            ghost = np.load(os.path.join(r"E:\PycharmProjects\GhostNet2\logs\GSDNet\GDD_hh\ghostnpy", img_name[:-4] + '.npy'))
            # print(ghost.shape)
            ghost = ghost.squeeze(0)
            ghost = Image.fromarray(ghost)

            w, h = img.size
            if img.mode != 'RGB':
                img = img.convert('RGB')

            input_data = Variable(img_transform(img).unsqueeze(0)).cuda()
            ghost_data = Variable(img_transform(ghost).unsqueeze(0)).cuda()

            final_glass = model(input_data, ghost_data)
            final_glass = final_glass[-1]

            final_glass = final_glass.data.squeeze(0)
            final_glass = np.array(transforms.Resize((h, w))(to_pil(final_glass)))

            Image.fromarray(final_glass).save(os.path.join(glass_path, img_name[:-4] + ".png"))

        end = time.time()
        print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))


if __name__ == '__main__':
    main()
